#!/usr/bin/python
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod, CRS, Transformer
import shapely
from shapely.geometry import MultiLineString, LineString
from .utils import get_pair_matrix, get_distance_matrix
g = Geod(ellps='WGS84')


def _get_closed_polygons(p_df, lons_bounds=None, lats_bounds=None):
    if lons_bounds is None:
        lons_bounds = [p_df.bounds.minx.min(), p_df.bounds.maxx.max()]

    if lats_bounds is None:
        lats_bounds = [p_df.bounds.miny.min(), p_df.bounds.maxy.max()]
    print('Corner points')
    print('Lons', lons_bounds, 'Lats', lats_bounds)

    corner_points = [[lons_bounds[0], lats_bounds[0]],
                     [lons_bounds[0], lats_bounds[1]],
                     [lons_bounds[1], lats_bounds[1]],
                     [lons_bounds[1], lats_bounds[0]],
                     [lons_bounds[0], lats_bounds[0]]]

    # Polygonize line segements
    lines = p_df.geometry.values.tolist()
    lines.append(LineString(corner_points))
    polygon = shapely.ops.unary_union(
        shapely.multipolygons(shapely.get_parts(shapely.polygonize(lines))))

    # Get multipolygons
    buffer = MultiLineString(lines).buffer(0.001)
    diff = polygon.difference(buffer)

    # Get new dataframe
    new_df = gpd.GeoDataFrame([], geometry=shapely.get_parts(diff))
    return new_df

# Find central plate


def _find_central_plate(plates_df, mean_lon=None, mean_lat=None):
    # Get the center lat lon of all plates
    if mean_lon is None:
        mean_lon = np.mean([plates_df.bounds.minx.min(),
                           plates_df.bounds.maxx.max()])

    if mean_lat is None:
        mean_lat = np.mean([plates_df.bounds.miny.min(),
                           plates_df.bounds.maxy.max()])

    centroids = np.vstack(plates_df.centroid.apply(lambda x: np.c_[x.xy]))

    # find the closes centroid to the center of scene
    ones = np.ones(centroids.shape[0])
    dist_from_mean = g.inv(ones * mean_lon, ones * mean_lat,
                           centroids[:, 0], centroids[:, 1])[2]

    index = np.where(dist_from_mean == np.min(dist_from_mean))[0]
    print(f'Central plate is with index: {index}')
    return index


def move_with_respect_to_plateix(data_df, plates_df, plate_ix, dist):
    df = data_df.sjoin(plates_df).groupby('index_right').apply(lambda x: x)
    # Get lat, lon from dataframe geometry
    df['lon'] = df.geometry.apply(lambda x: x.xy[0][0])
    df['lat'] = df.geometry.apply(lambda x: x.xy[1][0])
    df['mov_lon'] = None
    df['mov_lat'] = None

    for ix, _ in plates_df.iterrows():
        if ix != plate_ix:
            print(plate_ix, ix)
            mov_lon, mov_lat = move2plates(df.lon, df.lat,
                                           df, plate_ix,
                                           ix, dist)
        else:
            print(plate_ix, ix)
            mov_lon = df[df.index_right == ix].lon.values
            mov_lat = df[df.index_right == ix].lat.values

        df.loc[df.index_right == ix, 'mov_lat'] = mov_lat
        df.loc[df.index_right == ix, 'mov_lon'] = mov_lon

    return df


def move2plates(x, y, data_plate_df, ix1, ix2, dist):
    # ix1 is ref plate, dist in km
    df = gpd.GeoDataFrame(data_plate_df, geometry=gpd.points_from_xy(x, y))
    # df = data_df.sjoin(plates_df).groupby('index_right').apply(lambda k: k)
    # Get lat, lon from dataframe geometry
    df['mlon'] = df.geometry.apply(lambda k: k.xy[0][0])
    df['mlat'] = df.geometry.apply(lambda k: k.xy[1][0])

    # Get centroids
    hull1 = df[df.index_right == ix1].geometry.unary_union.convex_hull
    hull2 = df[df.index_right == ix2].geometry.unary_union.convex_hull

    az12, az21, _ = g.inv(hull1.centroid.xy[0],
                          hull1.centroid.xy[1],
                          hull2.centroid.xy[0],
                          hull2.centroid.xy[1])

    ones = np.ones(df[df.index_right == ix2].shape[0])
    mov_lon, mov_lat, _ = g.fwd(df[df.index_right == ix2].mlon,
                                df[df.index_right == ix2].mlat,
                                ones * az12, ones * dist)
    return mov_lon, mov_lat


def check_distance4moved_plate(df, dist, threshold=1):
    mov_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.mov_lon,
                                                              df.mov_lat))

    move_index = []
    mov_distance = []
    for ix1 in mov_df.index.unique(level=0):
        for ix2 in mov_df.index.unique(level=0):
            if ix2 > ix1:
                hull1 = mov_df[mov_df.index_right ==
                               ix1].geometry.unary_union.convex_hull
                hull2 = mov_df[mov_df.index_right ==
                               ix2].geometry.unary_union.convex_hull

                hull1_lonlat = np.vstack(hull1.exterior.xy)
                hull2_lonlat = np.vstack(hull2.exterior.xy)
                dist_matrix = get_distance_matrix(hull1_lonlat[0, :], hull1_lonlat[1, :],
                                                  hull2_lonlat[0, :], hull2_lonlat[1, :])
                if np.vstack(np.where(dist_matrix < dist)).size > 0:
                    mov_distance.append(
                        np.max(dist - dist_matrix[np.where(dist_matrix < dist)])+threshold)
                    move_index.append([ix1, ix2])

    return move_index, mov_distance


def change_coords_plate(gnss_df, xi, yi, plates_df, dist, central_ix=None):
    # GNSS input sites GeoDataFrame
    #sites = gpd.GeoDataFrame(
    #    gnss_df, geometry=gpd.points_from_xy(x, y), crs='EPSG:4326')
    #sites['site'] = site
    sites = gnss_df.copy()
    sites['grid_type'] = 'sites'
    
    # Interpolation grid GeoDataFrame
    grid = gpd.GeoDataFrame(
        [], geometry=gpd.points_from_xy(xi, yi), crs='EPSG:4326')
    grid['grid_type'] = 'grid'

    # Combine dataframes
    data = pd.concat([sites, grid])
    # Find central plate
    if central_ix is None:
        ix = _find_central_plate(plates_df,
                                 mean_lon=np.mean(xi),
                                 mean_lat=np.mean(yi))[0]
    else:
        ix = np.squeeze(central_ix)#[0]
    print(ix)
    # Move with respect to the central plate
    mov_df = move_with_respect_to_plateix(data, plates_df, ix, dist*1e3)

    # Repeat moving plate as long as all points are not moved with specified distance
    print('Work again')
    print('ix1 ix2 Distance')
    ixs, dists = check_distance4moved_plate(mov_df, dist)
    while len(ixs) > 1:
        for i, d in zip(ixs, dists):
            if ix in i:
                i = np.array(i)
                ix2 = i[i != ix][0]
                print(ix, ix2, d)
                lon, lat = move2plates(
                    mov_df.mov_lon, mov_df.mov_lat, mov_df, ix, ix2, np.rint(d)*1e3)
                mov_df.loc[mov_df.index_right == ix2, 'mov_lat'] = lat
                mov_df.loc[mov_df.index_right == ix2, 'mov_lon'] = lon
            else:
                ix1, ix2 = i
                print(ix1, ix2, np.rint(d))
                lon, lat = move2plates(
                    mov_df.mov_lon, mov_df.mov_lat, mov_df, ix1, ix2, np.rint(d)*1e3)
                mov_df.loc[mov_df.index_right == ix2, 'mov_lat'] = lat
                mov_df.loc[mov_df.index_right == ix2, 'mov_lon'] = lon
            ixs, dists = check_distance4moved_plate(mov_df, dist)
    return mov_df
