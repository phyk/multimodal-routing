import overpass
from shapely.geometry import MultiPolygon, Polygon


def fetch_boundary_polygon(city_name_german: str, admin_level: int) -> Polygon:
    """
    Fetches the boundary polygon for a given city and administrative level using the Overpass API.

    :param city_name_german: str - The name of the city in German to query.
    :param admin_level: int - The administrative level for the boundary (e.g., 6 for Koeln, 4 for Berlin).
    :returns: Polygon - A Shapely Polygon object representing the boundary of the specified city.
    """
    query = f"""area["name"="{city_name_german}"]->.searchArea;
    relation["boundary"="administrative"]["admin_level"="{admin_level}"](area.searchArea);"""
    api = overpass.API()
    result = api.get(query, verbosity="geom", responseformat="geojson")
    biggest_polygon = []
    for [polygon] in result["features"][0]["geometry"]["coordinates"]:
        if len(polygon) > len(biggest_polygon):
            biggest_polygon = polygon

    boundary_polygon_ = MultiPolygon(result["features"][0]["geometry"]["coordinates"])

    largest_polygon = boundary_polygon_.geoms[0]
    for polygon in boundary_polygon_.geoms:
        if polygon.area > largest_polygon.area:
            largest_polygon = polygon

    return largest_polygon


if __name__ == "__main__":
    _ = fetch_boundary_polygon("Köln", 6)
