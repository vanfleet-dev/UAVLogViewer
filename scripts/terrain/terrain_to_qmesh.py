#!/usr/bin/env python3
"""Prepare a regional quantized-mesh package from USGS 3DEP 1 m or a local raster.

The USGS path queries The National Map for a bounding box, keeps the newest
GeoTIFF for each 10 km tile, downloads it, records provenance and checksums,
reprojects/mosaics the source to EPSG:4326, and uses the existing ArduPilot
quantized-mesh encoder. Source elevation values are preserved without vertical
datum conversion.
"""

import argparse
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlencode, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

import srtm_to_qmesh as qmesh

TNM_PRODUCTS_URL = 'https://tnmaccess.nationalmap.gov/api/v1/products'
USGS_1M_DATASET = 'Digital Elevation Model (DEM) 1 meter'
USGS_ELEVATION_REFERENCE = 'NAVD88 metres as supplied by USGS; no vertical conversion applied'
TILE_NAME_RE = re.compile(r'\bx(\d+)y(\d+)\b', re.IGNORECASE)


def validate_bbox(bbox):
    west, south, east, north = bbox
    if not (-180.0 <= west < east <= 180.0):
        raise ValueError('bbox must satisfy -180 <= WEST < EAST <= 180')
    if not (-90.0 <= south < north <= 90.0):
        raise ValueError('bbox must satisfy -90 <= SOUTH < NORTH <= 90')
    return [float(value) for value in bbox]


def product_key(product):
    match = TILE_NAME_RE.search(product.get('title', ''))
    if match:
        return 'tile', int(match.group(1)), int(match.group(2))
    bounds = product.get('boundingBox') or {}
    return ('bounds',
            round(float(bounds.get('minX', 0.0)), 3),
            round(float(bounds.get('minY', 0.0)), 3),
            round(float(bounds.get('maxX', 0.0)), 3),
            round(float(bounds.get('maxY', 0.0)), 3))


def select_current_products(products, probe=None):
    """Keep the newest usable product for each USGS 10 km tile."""
    grouped = {}
    for product in products:
        url = product.get('downloadURL')
        if not url or not url.lower().split('?', 1)[0].endswith(('.tif', '.tiff')):
            continue
        key = product_key(product)
        grouped.setdefault(key, []).append(product)
    selected = []
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key], key=lambda item: item.get('publicationDate', ''),
            reverse=True)
        if probe is None:
            selected.append(candidates[0])
            continue
        for candidate in candidates:
            if probe(candidate):
                selected.append(candidate)
                break
    return selected


def product_has_data(product, bbox):
    """Probe the requested overlap through GDAL HTTP ranges before downloading."""
    product_bounds = product.get('boundingBox') or {}
    west = max(bbox[0], float(product_bounds.get('minX', bbox[0])))
    south = max(bbox[1], float(product_bounds.get('minY', bbox[1])))
    east = min(bbox[2], float(product_bounds.get('maxX', bbox[2])))
    north = min(bbox[3], float(product_bounds.get('maxY', bbox[3])))
    if west >= east or south >= north:
        return False
    points = []
    for fy in (0.2, 0.5, 0.8):
        for fx in (0.2, 0.5, 0.8):
            points.append((west + fx * (east - west),
                           south + fy * (north - south)))
    try:
        with rasterio.open(product['downloadURL']) as source:
            from rasterio.warp import transform
            xs, ys = transform(
                'EPSG:4326', source.crs,
                [point[0] for point in points],
                [point[1] for point in points])
            usable = 0
            for value in source.sample(zip(xs, ys), masked=True):
                height = value[0]
                if np.ma.is_masked(height) or not np.isfinite(height):
                    continue
                if source.nodata is not None and height == source.nodata:
                    continue
                usable += 1
    except Exception as error:
        print('probe failed for %s: %s' % (product.get('title', 'product'), error),
              flush=True)
        return False
    if usable == len(points):
        print('use %s (%d/9 probe samples valid)' %
              (product.get('title', 'product'), usable), flush=True)
    elif usable:
        print('skip partial %s (%d/9 probe samples valid)' %
              (product.get('title', 'product'), usable), flush=True)
    else:
        print('skip empty %s' % product.get('title', 'product'), flush=True)
    return usable == len(points)


def query_usgs_products(bbox):
    all_items = []
    offset = 0
    page_size = 500
    while True:
        params = {
            'bbox': ','.join(str(value) for value in bbox),
            'datasets': USGS_1M_DATASET,
            'prodFormats': 'GeoTIFF',
            'outputFormat': 'JSON',
            'max': page_size,
            'offset': offset,
        }
        url = '%s?%s' % (TNM_PRODUCTS_URL, urlencode(params))
        request = Request(url, headers={'User-Agent': 'MAVProxy-terrain-prep/1'})
        with urlopen(request, timeout=60) as response:
            page = json.load(response)
        items = page.get('items', [])
        all_items.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get('total', len(all_items))):
            break
    return select_current_products(
        all_items, probe=lambda product: product_has_data(product, bbox)), url


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_product(product, download_dir, force=False):
    url = product['downloadURL']
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        raise RuntimeError('USGS product has no filename: %s' % url)
    path = download_dir / name
    expected_size = int(product.get('sizeInBytes') or 0)
    if path.exists() and not force and (not expected_size or path.stat().st_size == expected_size):
        print('reuse %s' % path, flush=True)
    else:
        part = path.with_suffix(path.suffix + '.part')
        part.unlink(missing_ok=True)
        print('download %s' % url, flush=True)
        request = Request(url, headers={'User-Agent': 'MAVProxy-terrain-prep/1'})
        with urlopen(request, timeout=120) as response, open(part, 'wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if expected_size and part.stat().st_size != expected_size:
            part.unlink(missing_ok=True)
            raise RuntimeError('download size mismatch for %s' % url)
        os.replace(part, path)
    record = dict(product)
    record['localFile'] = path.name
    record['downloadedBytes'] = path.stat().st_size
    record['sha256'] = sha256_file(path)
    return path, record


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write('\n')
    os.replace(tmp, path)


def raster_record(path):
    with rasterio.open(path) as source:
        return {
            'file': path.name,
            'crs': source.crs.to_string() if source.crs else None,
            'bounds': list(source.bounds),
            'resolution': [abs(source.res[0]), abs(source.res[1])],
            'width': source.width,
            'height': source.height,
            'dtype': source.dtypes[0],
            'nodata': source.nodata,
        }


def build_mosaic(source_paths, destination, bounds):
    destination.unlink(missing_ok=True)
    records = [raster_record(path) for path in source_paths]
    with ExitStack() as stack:
        warped = []
        for path in source_paths:
            source = stack.enter_context(rasterio.open(path))
            if source.crs is None:
                raise RuntimeError('%s has no coordinate reference system' % path)
            warped.append(stack.enter_context(WarpedVRT(
                source, crs='EPSG:4326', resampling=Resampling.bilinear)))
        merge(
            warped,
            bounds=bounds,
            dtype='float32',
            nodata=-32768.0,
            resampling=Resampling.bilinear,
            use_highest_res=True,
            mem_limit=128,
            dst_path=destination,
            dst_kwds={
                'driver': 'GTiff',
                'compress': 'deflate',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'BIGTIFF': 'IF_SAFER',
            })
    return records, raster_record(destination)


def mosaic_valid_fraction(path, bbox):
    with rasterio.open(path) as source:
        window = rasterio.windows.from_bounds(*bbox, transform=source.transform)
        sample = source.read(
            1, window=window, out_shape=(128, 128),
            resampling=Resampling.nearest, boundless=True, masked=True)
    return float(np.count_nonzero(~np.ma.getmaskarray(sample))) / sample.size


def render_tiles(source_path, out_dir, bbox, minzoom, maxzoom, jobs, grid,
                 max_error, tile_px, force):
    keys, _expected = qmesh.build_worklist_for_bbox(
        bbox, minzoom, maxzoom, want_levels=False)
    total = len(keys)
    if total == 0:
        raise RuntimeError('bbox produced no terrain tiles')
    print('tiles=%d zoom=%d-%d jobs=%d' % (total, minzoom, maxzoom, jobs), flush=True)
    per_level = {zoom: [] for zoom in range(minzoom, maxzoom + 1)}
    counts = {'ok': 0, 'skip': 0, 'empty': 0}
    started = time.time()
    with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=qmesh._init_worker,
            initargs=(str(source_path), str(out_dir), max_error, tile_px,
                      force, None, -1, grid)) as executor:
        batch_size = 100000
        done = 0
        for start in range(0, total, batch_size):
            batch = [qmesh._decode_key(key)
                     for key in keys[start:start + batch_size].tolist()]
            for tile, result in zip(batch, executor.map(qmesh.render_tile, batch, chunksize=16)):
                counts[result] = counts.get(result, 0) + 1
                done += 1
                if result in ('ok', 'skip'):
                    zoom, x, y = tile
                    per_level[zoom].append((x, y))
                if done % 100 == 0 or done == total:
                    print('  %d/%d ok=%d skip=%d empty=%d' %
                          (done, total, counts['ok'], counts['skip'], counts['empty']),
                          flush=True)
    if counts['ok'] + counts['skip'] == 0:
        raise RuntimeError('source produced no terrain tiles')
    print('rendered in %.1fs' % (time.time() - started), flush=True)
    return per_level, counts


def prepare(args):
    bbox = validate_bbox(args.bbox)
    if args.min_zoom < 0 or args.max_zoom > 19 or args.min_zoom > args.max_zoom:
        raise ValueError('zoom range must satisfy 0 <= MIN <= MAX <= 19')

    package_dir = Path(args.out).resolve()
    terrain_dir = package_dir / 'terrain'
    package_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(exist_ok=True)
    incomplete = package_dir / '.incomplete'
    incomplete.write_text('terrain package build in progress\n')

    manifest = {
        'schemaVersion': 1,
        'builtAt': datetime.now(timezone.utc).isoformat(),
        'converter': {
            'file': Path(__file__).name,
            'sha256': sha256_file(Path(__file__)),
        },
        'dataset': USGS_1M_DATASET if args.usgs_1m else 'local raster',
        'operationalAOI': bbox,
        'elevationReference': (USGS_ELEVATION_REFERENCE if args.usgs_1m
                               else args.elevation_reference),
        'verticalValuesTransformed': False,
        'nodataPolicy': 'refuse package when sampled AOI validity is below 99 percent',
        'productsQuery': None,
        'products': [],
        'sourceRasters': [],
    }

    if args.usgs_1m:
        products, query_url = query_usgs_products(bbox)
        if not products:
            raise RuntimeError('USGS has no 1 m GeoTIFF products for this bbox')
        if len(products) > args.max_products:
            raise RuntimeError('%d USGS tiles exceed --max-products %d' %
                               (len(products), args.max_products))
        print('USGS products selected: %d' % len(products), flush=True)
        download_dir = package_dir / 'sources'
        download_dir.mkdir(exist_ok=True)
        source_paths = []
        records = []
        for product in products:
            path, record = download_product(product, download_dir, args.force_download)
            source_paths.append(path)
            records.append(record)
        manifest['productsQuery'] = query_url
        manifest['products'] = records
    else:
        source_path = Path(args.source_raster).resolve()
        if not source_path.is_file():
            raise RuntimeError('source raster not found: %s' % source_path)
        source_paths = [source_path]

    x0, x1, y0, y1 = qmesh.tile_range_for_bbox(args.max_zoom, *bbox)
    first_bounds = qmesh.tile_bounds(args.max_zoom, x0, y0)
    last_bounds = qmesh.tile_bounds(args.max_zoom, x1, y1)
    mosaic_bounds = [first_bounds[0], first_bounds[1],
                     last_bounds[2], last_bounds[3]]
    manifest['preparedBounds'] = mosaic_bounds
    mosaic_path = package_dir / '_source.tif'
    source_records, mosaic_record = build_mosaic(
        source_paths, mosaic_path, mosaic_bounds)
    manifest['sourceRasters'] = source_records
    manifest['mosaic'] = mosaic_record
    valid_fraction = mosaic_valid_fraction(mosaic_path, bbox)
    manifest['mosaic']['requestedBoundsValidFraction'] = valid_fraction
    if valid_fraction < 0.99:
        atomic_json(package_dir / 'region.json', manifest)
        raise RuntimeError(
            'source data covers only %.1f%% of the requested bbox; choose a smaller '
            'zone or another source raster' % (100.0 * valid_fraction))
    per_level, counts = render_tiles(
        mosaic_path, terrain_dir, bbox, args.min_zoom, args.max_zoom,
        args.jobs, args.grid, args.max_error, args.tile_px, args.force)
    manifest['terrain'] = {
        'path': 'terrain/layer.json',
        'minZoom': args.min_zoom,
        'maxZoom': args.max_zoom,
        'tileCounts': {str(zoom): len(per_level[zoom]) for zoom in per_level},
        'renderCounts': counts,
    }
    atomic_json(package_dir / 'region.json', manifest)
    metadata = {
        'dataset': manifest['dataset'],
        'elevationReference': manifest['elevationReference'],
        'sourceManifest': '../region.json',
        'sourceCRS': [record['crs'] for record in source_records],
        'tileCounts': {str(zoom): len(per_level[zoom]) for zoom in per_level},
    }
    qmesh.write_layer_json(
        terrain_dir, args.max_zoom, per_level, args.min_zoom,
        bounds=bbox, output_minzoom=args.min_zoom, metadata=metadata)
    incomplete.unlink(missing_ok=True)
    print('complete: %s (%s)' % (package_dir, counts), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--usgs-1m', action='store_true',
                        help='query and download current USGS 3DEP 1 m GeoTIFFs')
    source.add_argument('--source-raster', help='local georeferenced DEM raster')
    parser.add_argument('--bbox', nargs=4, required=True, type=float,
                        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    parser.add_argument('--out', required=True, help='output terrain package directory')
    parser.add_argument('--min-zoom', type=int, default=13)
    parser.add_argument('--max-zoom', type=int, default=18)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--grid', type=int, default=65)
    parser.add_argument('--tile-px', type=int, default=256)
    parser.add_argument('--max-error', type=float, default=4.0)
    parser.add_argument('--max-products', type=int, default=100,
                        help='abort before downloading more than this many USGS tiles')
    parser.add_argument('--elevation-reference', default='source values; no vertical conversion')
    parser.add_argument('--force', action='store_true', help='rebuild existing terrain tiles')
    parser.add_argument('--force-download', action='store_true',
                        help='redownload existing source GeoTIFFs')
    return parser.parse_args(argv)


def main():
    try:
        prepare(parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit('terrain preparation failed: %s' % error)


if __name__ == '__main__':
    main()
