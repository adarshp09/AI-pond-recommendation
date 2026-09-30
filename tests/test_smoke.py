import io, zipfile
import numpy as np
from fastapi.testclient import TestClient
from main import app
from backend.parser import parse_contour_file
from backend.terrain import calculate_slope, calculate_d8_flow, calculate_flow_accumulation

client=TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"

def test_flat_slope():
    assert np.allclose(calculate_slope(np.ones((5,5)),1),0)

def test_d8():
    dem=np.array([[5.,4.,3.],[6.,5.,2.],[7.,6.,1.]])
    _,down=calculate_d8_flow(dem,1)
    acc=calculate_flow_accumulation(down)
    assert acc.shape==dem.shape
    assert np.isfinite(acc).all()

def test_kml():
    kml=b"""<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>100</name><LineString><coordinates>81,21 81.001,21</coordinates></LineString></Placemark></Document></kml>"""
    p=parse_contour_file(kml,"test.kml")
    assert len(p.contours)==1
    assert p.contours[0].elevation_m==100

def test_kmz():
    kml=b"""<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>100</name><LineString><coordinates>81,21 81.001,21</coordinates></LineString></Placemark></Document></kml>"""
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z: z.writestr("doc.kml",kml)
    p=parse_contour_file(b.getvalue(),"test.kmz")
    assert len(p.contours)==1
