#!/usr/bin/python
import numpy as np
from pyproj import Geod
from shapely.geometry import MultiPoint, MultiLineString, LineString


def get_pair_matrix(data1, data2):
    if data1.ndim > 1 or data2.ndim > 1:
        raise ValueError('Input arrays need to be 1D array')

    n_data1 = data1.shape[0]
    n_data2 = data2.shape[0]

    data1 = np.repeat(np.atleast_2d(data1), n_data2, axis=0)
    data2 = np.repeat(np.atleast_2d(data2).T, n_data1, axis=1)

    return data1, data2


def get_distance_matrix(lon1, lat1, lon2, lat2, unit='km'):
    # Output numpy array of distances in km
    g = Geod(ellps='WGS84')
    lon1, lon2 = get_pair_matrix(lon1, lon2)
    lat1, lat2 = get_pair_matrix(lat1, lat2)
    # pyproj gives distance in meters
    scale = {'cm': 100., 'dm': 10., 'm': 1, 'km': 1e-3}

    return g.inv(lon1, lat1, lon2, lat2)[2] * scale[unit]


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_intersection_mask(lon, lat, intersection_geometry):
    lon1, lon2 = get_pair_matrix(lon, lon)
    lat1, lat2 = get_pair_matrix(lat, lat)

    # Get points
    points = MultiPoint(np.c_[lon, lat])

    # Get all possible connections
    connections = []

    for i in range(len(points.geoms)):
        for j in range(i + 1, len(points.geoms)):
            line = LineString([points.geoms[i], points.geoms[j]])
            connections.append([i, j, line])

    connections = MultiLineString(np.vstack(connections)[:, 2].tolist())

    # Find intersections
    dist_mask = np.zeros((len(points.geoms), len(points.geoms)))
    for line in connections.geoms:
        flag = line.intersects(intersection_geometry)

        if flag is True:
            ix1 = (line.xy[0][0] == lon1) * (line.xy[0][1] == lon2)
            ix2 = (line.xy[1][0] == lat1) * (line.xy[1][1] == lat2)
            dist_mask[(ix1*ix2)] = 1

    # Get full mask
    dist_mask = dist_mask.T + dist_mask

    return dist_mask


def get_intersection_mask2(lon1, lat1, lon2, lat2,
                           intersection_geometry, threads=4):
    import multiprocessing
    lon_1, lon_2 = get_pair_matrix(lon2, lon1)
    lat_1, lat_2 = get_pair_matrix(lat2, lat1)

    # Get points
    points1 = MultiPoint(np.c_[lon1, lat1])
    points2 = MultiPoint(np.c_[lon2, lat2])

    # Find intersections
    dist_mask = np.zeros((len(points1.geoms), len(points2.geoms)))

    def check_intersects(p1, p2):
        if p1 != p2:
            line = LineString([p1, p2])
            flag = line.intersects(intersection_geometry)
            if flag is True:
                ix1 = (line.xy[0][0] == lon_1) * (line.xy[0][1] == lon_2)
                ix2 = (line.xy[1][0] == lat_1) * (line.xy[1][1] == lat_2)
                dist_mask[(ix1*ix2)] = 1

    def loop_point(p1):
        def ch_intrs(x):
            return check_intersects(p1, x)
        with multiprocessing.pool.ThreadPool(threads) as pool:
            pool.map(ch_intrs, points2.geoms)

    with multiprocessing.pool.ThreadPool(threads) as pool:
        pool.map(loop_point, points1.geoms)

    return dist_mask
