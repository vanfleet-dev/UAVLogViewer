import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import srtm_to_qmesh as qmesh
import terrain_to_qmesh as terrain


def test_bbox_worklist_contains_only_intersecting_tiles():
    bbox = [149.0000, -35.9995, 149.0010, -35.9985]
    keys, per_level = qmesh.build_worklist_for_bbox(bbox, 17, 18)

    assert len(keys) > 0
    assert set(per_level) == {17, 18}
    for packed in keys:
        z, x, y = qmesh._decode_key(packed)
        west, south, east, north = qmesh.tile_bounds(z, x, y)
        assert west < bbox[2]
        assert east > bbox[0]
        assert south < bbox[3]
        assert north > bbox[1]


def test_layer_json_records_regional_contract(tmp_path):
    bbox = [-105.276, 39.994, -105.274, 39.996]
    _, per_level = qmesh.build_worklist_for_bbox(bbox, 17, 18)
    metadata = {
        'dataset': 'Digital Elevation Model (DEM) 1 meter',
        'elevationReference': 'NAVD88 as supplied by USGS',
    }

    qmesh.write_layer_json(
        tmp_path, 18, per_level, 17, bounds=bbox,
        output_minzoom=17, metadata=metadata)

    layer = json.loads((tmp_path / 'layer.json').read_text())
    assert layer['bounds'] == bbox
    assert layer['minzoom'] == 17
    assert layer['maxzoom'] == 18
    assert layer['metadata'] == metadata
    assert layer['available'][17]
    assert layer['available'][18]


def test_regional_layer_does_not_merge_stale_availability(tmp_path):
    qmesh.write_layer_json(
        str(tmp_path), 18, {18: [(10, 20)]}, 18,
        bounds=[1.0, 2.0, 3.0, 4.0], output_minzoom=18,
        merge_existing=False)
    qmesh.write_layer_json(
        str(tmp_path), 18, {18: [(30, 40)]}, 18,
        bounds=[5.0, 6.0, 7.0, 8.0], output_minzoom=18,
        merge_existing=False)

    layer = json.loads((tmp_path / 'layer.json').read_text())

    assert layer['bounds'] == [5.0, 6.0, 7.0, 8.0]
    assert layer['available'][18] == [
        {'startX': 30, 'startY': 40, 'endX': 30, 'endY': 40}]


def test_latest_product_is_selected_for_each_usgs_tile():
    products = [
        _product('old x47y443', '2022-02-10', 'https://example/old.tif', 10),
        _product('new x47y443', '2026-03-20', 'https://example/new.tif', 11),
        _product('only x48y443', '2024-01-01', 'https://example/other.tif', 12),
    ]

    selected = terrain.select_current_products(products)

    assert [item['downloadURL'] for item in selected] == [
        'https://example/new.tif',
        'https://example/other.tif',
    ]


def test_product_selection_falls_back_when_newest_has_no_data():
    products = [
        _product('old x47y443', '2022-02-10', 'https://example/old.tif', 10),
        _product('new x47y443', '2026-03-20', 'https://example/new.tif', 11),
    ]

    selected = terrain.select_current_products(
        products, probe=lambda product: 'old.tif' in product['downloadURL'])

    assert [item['downloadURL'] for item in selected] == ['https://example/old.tif']


def test_rejects_invalid_bbox_order():
    with pytest.raises(ValueError, match='WEST < EAST'):
        terrain.validate_bbox([-105.0, 40.0, -106.0, 41.0])


def test_lod_worklist_matches_default_map3d_footprint():
    bbox = [-105.2751, 39.9949, -105.2750, 39.9950]

    keys, per_level, focus_count = terrain.build_lod_worklist(
        bbox, 13, 18, fine_radius=2, ring=1, max_focus_tiles=100)

    assert focus_count == 1
    assert len(keys) == 130
    assert [len(per_level[zoom]) for zoom in range(13, 19)] == [21, 21, 21, 21, 21, 25]
    for zoom in (13, 18):
        xs = [x for x, _y in per_level[zoom]]
        ys = [y for _x, y in per_level[zoom]]
        assert max(xs) - min(xs) + 1 == 5
        assert max(ys) - min(ys) + 1 == 5


def test_lod_worklist_does_not_fill_coarse_bounds_at_fine_zoom():
    bbox = list(qmesh.tile_bounds(18, 108825, 189318))
    bbox[2] = qmesh.tile_bounds(18, 108826, 189318)[2]

    keys, per_level, _focus_count = terrain.build_lod_worklist(
        bbox, 13, 18, fine_radius=2, ring=1, max_focus_tiles=100)
    prepared = terrain.worklist_bounds(keys)
    x0, x1, y0, y1 = qmesh.tile_range_for_bbox(18, *prepared)
    full_rectangle_count = (x1 - x0 + 1) * (y1 - y0 + 1)

    assert len(per_level[18]) < full_rectangle_count // 100


def test_local_raster_build_publishes_regional_package(tmp_path):
    source = tmp_path / 'source.tif'
    bbox = [-105.001, 39.999, -104.999, 40.001]
    x0, _x1, y0, _y1 = qmesh.tile_range_for_bbox(13, *bbox)
    source_bounds = terrain.sampling_bounds(
        qmesh.tile_bounds(13, x0, y0), 13, 65)
    with rasterio.open(
            source, 'w', driver='GTiff', width=256, height=256, count=1,
            dtype='float32', crs='EPSG:4326', nodata=-32768.0,
            transform=from_bounds(*source_bounds, 256, 256)) as dataset:
        dataset.write(np.full((256, 256), 1234.5, dtype=np.float32), 1)
    package = tmp_path / 'package'
    args = terrain.parse_args([
        '--source-raster', str(source),
        '--bbox', *(str(value) for value in bbox),
        '--min-zoom', '13', '--max-zoom', '13',
        '--out', str(package), '--jobs', '1',
        '--fine-radius', '0', '--lod-ring', '0',
    ])

    terrain.prepare(args)

    region = json.loads((package / 'region.json').read_text())
    assert region['schemaVersion'] == 1
    assert region['terrain']['path'] == 'terrain/layer.json'
    assert region['terrain']['renderCounts']['ok'] == 1
    assert region['lodPolicy']['fineRadius'] == 0
    assert region['lodPolicy']['localHandoffZoom'] == 13
    assert (package / 'terrain' / 'layer.json').is_file()
    assert len(list((package / 'terrain' / '13').glob('*/*.terrain'))) == 1
    assert not (package / '.incomplete').exists()


def _product(title, publication_date, url, size):
    return {
        'title': title,
        'publicationDate': publication_date,
        'downloadURL': url,
        'sizeInBytes': size,
        'boundingBox': {
            'minX': -105.4,
            'minY': 39.9,
            'maxX': -105.2,
            'maxY': 40.1,
        },
    }
