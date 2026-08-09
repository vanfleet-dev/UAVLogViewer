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


def test_local_raster_build_publishes_regional_package(tmp_path):
    source = tmp_path / 'source.tif'
    bbox = [-105.001, 39.999, -104.999, 40.001]
    with rasterio.open(
            source, 'w', driver='GTiff', width=256, height=256, count=1,
            dtype='float32', crs='EPSG:4326', nodata=-32768.0,
            transform=from_bounds(*bbox, 256, 256)) as dataset:
        dataset.write(np.full((256, 256), 1234.5, dtype=np.float32), 1)
    package = tmp_path / 'package'
    args = terrain.parse_args([
        '--source-raster', str(source),
        '--bbox', *(str(value) for value in bbox),
        '--min-zoom', '13', '--max-zoom', '13',
        '--out', str(package), '--jobs', '1',
    ])

    terrain.prepare(args)

    region = json.loads((package / 'region.json').read_text())
    assert region['schemaVersion'] == 1
    assert region['terrain']['path'] == 'terrain/layer.json'
    assert region['terrain']['renderCounts']['ok'] == 1
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
