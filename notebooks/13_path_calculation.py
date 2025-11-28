import logging
import pathlib
import pickle
import tomllib

import mcr_py.helper_functions
import mcr_py.mcr.path
import mcr_py.utils.strtime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from mcr_py.mcr.data import NetworkType, OSMData
from mcr_py.mcr.label import IntermediateLabel
from mcr_py.mcr.path import GTFSPath, Path, PathType
from mcr_py.utils.logger import setup


def plot_paths_on_map(
    labels: pd.DataFrame,
    nodes: pd.DataFrame,
    path_manager: mcr_py.mcr.path.PathManager,
    translator_map: dict[PathType, dict[str, int]],
    color_map: dict[str, str],
    path_name: pathlib.Path,
    stops_by_id: pd.DataFrame,
) -> None:
    logging.info("Generating path image %s", path_name)
    nodes_by_id = nodes.set_index("id", drop=False)
    fig = go.Figure()

    walking_paths_long = []
    walking_paths_lat = []
    cycling_paths_long = []
    cycling_paths_lat = []
    pt_paths_long = []
    pt_paths_lat = []
    poi_assoc_long = []
    poi_assoc_lat = []

    for row in labels.itertuples():
        label: IntermediateLabel = row.label  # type: ignore
        end_node_id = row.osm_node_id  # type: ignore
        end_node = nodes_by_id.loc[end_node_id]

        paths = mcr_py.mcr.path.reconstruct_and_translate_path_for_label(
            path_manager.paths, label, translator_map
        )
        for i, path in enumerate(paths):
            if isinstance(path, Path):
                if path.path == []:
                    continue
                if path.path_type == PathType.WALKING:
                    walking_path_nodes = [nodes_by_id.loc[node_id] for node_id in path.path]
                    path_lat = [node.lat for node in walking_path_nodes]
                    path_lon = [node.long for node in walking_path_nodes]
                    if i + 1 == len(paths):
                        path_lat.append(end_node.lat)  # type: ignore
                        path_lon.append(end_node.long)  # type: ignore
                    else:
                        if isinstance(paths[i + 1], GTFSPath):
                            next_start_stop = stops_by_id.loc[str(paths[i + 1].start_stop_id)]
                            path_lat.append(next_start_stop.stop_lat)
                            path_lon.append(next_start_stop.stop_lon)
                        else:
                            next_start_node = nodes_by_id.loc[int(paths[i + 1].path[0][1:])]
                            path_lat.append(next_start_node.lat)
                            path_lon.append(next_start_node.long)
                    walking_paths_long.extend(path_lon)
                    walking_paths_long.append(None)
                    walking_paths_lat.extend(path_lat)
                    walking_paths_lat.append(None)
                if path.path_type in [PathType.DRIVING_WALKING, PathType.CYCLING_WALKING]:
                    path_nodes = [
                        nodes_by_id.loc[int(node_id[1:])]  # pyright: ignore[reportIndexIssue]
                        for node_id in path.path
                        if node_id[0] == "D"  # pyright: ignore[reportIndexIssue]
                    ]
                    path_lat = [node.lat for node in path_nodes]
                    path_lon = [node.long for node in path_nodes]
                    cycling_paths_long.extend(path_lon)
                    cycling_paths_long.append(None)
                    cycling_paths_lat.extend(path_lat)
                    cycling_paths_lat.append(None)
            elif isinstance(path, GTFSPath):
                start_stop_id = path.start_stop_id
                end_stop_id = path.end_stop_id
                start_stop = stops_by_id.loc[str(start_stop_id)]
                end_stop = stops_by_id.loc[str(end_stop_id)]
                trip = path.trip_id
                if len(trip) >= 10:
                    trip = trip[:10] + "..."

                path_lat = [node.stop_lat for node in [start_stop, end_stop]]
                path_lon = [node.stop_lon for node in [start_stop, end_stop]]
                pt_paths_long.extend(path_lon)
                pt_paths_long.append(None)
                pt_paths_lat.extend(path_lat)
                pt_paths_lat.append(None)
        if row.poi_type is not np.nan:  # type: ignore
            poi_assoc_long.append(row.poi_long)  # type: ignore
            poi_assoc_long.append(end_node.long)
            poi_assoc_lat.append(row.poi_lat)  # type: ignore
            poi_assoc_lat.append(end_node.lat)
            poi_assoc_long.append(None)
            poi_assoc_lat.append(None)

    fig.add_trace(
        go.Scattermap(
            lon=walking_paths_long,
            lat=walking_paths_lat,
            mode="lines",
            marker={"size": 1, "color": "grey"},
            legendgroup="Walking",
            name="Walking",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=cycling_paths_long,
            lat=cycling_paths_lat,
            mode="lines",
            marker={"size": 1, "color": "blue"},
            legendgroup="Bicycle",
            name="Bicycle",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=pt_paths_long,
            lat=pt_paths_lat,
            mode="lines",
            marker={"size": 1, "color": "green"},
            legendgroup="Public Transport",
            name="Public Transport",
            showlegend=True,
        )
    )

    nodes_long = [nodes_by_id.loc[row.osm_node_id].long for row in labels.itertuples()]  # type: ignore
    nodes_lat = [nodes_by_id.loc[row.osm_node_id].lat for row in labels.itertuples()]  # type: ignore
    fig.add_trace(
        go.Scattermap(
            lon=nodes_long,
            lat=nodes_lat,
            mode="markers",
            marker={"size": 8, "color": "grey"},
            name="Node",
            legendgroup="Nodes",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scattermap(
            lon=poi_assoc_long,
            lat=poi_assoc_lat,
            mode="lines",
            marker={"size": 8, "color": "grey"},  # pyright: ignore[reportArgumentType]
            showlegend=False,
        )
    )

    for poi_type in labels.poi_type.dropna().unique():
        labels_with_poi_type = labels[labels["poi_type"] == poi_type]
        poi_long = [
            row.poi_long  # type: ignore
            for row in labels_with_poi_type.itertuples()
            if row.poi_type == poi_type  # type: ignore
        ]
        poi_lat = [
            row.poi_lat  # type: ignore
            for row in labels_with_poi_type.itertuples()
            if row.poi_type == poi_type  # type: ignore
        ]
        fig.add_trace(
            go.Scattermap(
                lon=poi_long,
                lat=poi_lat,
                mode="markers",
                marker={"size": 10, "color": color_map[poi_type]},
                name=poi_type,
                legendgroup=poi_type,
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scattermap(
            lat=["50.948884"],  # Latitude of the marker
            lon=["6.917342"],  # Longitude of the marker
            mode="markers",
            marker={"size": 14, "color": "lightgreen"},
            text=["Starting Point"],  # Hover text
            name="Starting Point",
            showlegend=True,
            legendgroup="Starting Point",
        )
    )

    fig.update_layout(
        map={
            "style": "carto-positron-nolabels",
            "zoom": 16,  # street level
            "center": {"lat": 50.948884, "lon": 6.917342},
        },
        legend={
            "title": {
                "text": "Transport & POI Types",  # your legend title
                "font": {"size": 14, "color": "black"},
            },
            "orientation": "v",  # vertical (default) or "h" for horizontal
            "x": 1,  # horizontal position (0=left, 1=right)
            "y": 1,  # vertical position (0=bottom, 1=top)
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=2000,
        width=2400,
    )
    fig.write_image(path_name, scale=2)


def generate_path_image(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    geo_data: OSMData,
    gtfs_clean_stops: pathlib.Path,
) -> None:
    with input_path.open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    labels = pd.DataFrame(
        [
            (label.node_id, label.values[0], label.values[1], n_transfers, label)
            for n_transfers, bags in data["bags_i"].items()
            for bag in bags.values()
            for label in bag
        ],
        columns=["osm_node_id", "time", "cost", "n_transfers", "label"],  # type: ignore
    )
    labels["human_time"] = labels["time"].apply(
        lambda x: mcr_py.utils.strtime.seconds_to_str_time(x, 10)
    )
    labels = labels.merge(
        geo_data.pois.select(
            pl.col("nearest_osm_node").alias("osm_node_id").cast(pl.Int64),
            pl.col("lat").alias("poi_lat"),
            pl.col("long").alias("poi_long"),
            "poi_type",
        ).to_pandas(),
        how="left",
        on="osm_node_id",
    )
    nodes = geo_data.osm_nodes.with_columns(pl.col("osm_id").alias("id")).to_pandas()
    if "car" in output_path.name:
        nodes = (
            geo_data.additional_networks[NetworkType.DRIVING][0]
            .with_columns(pl.col("osm_id").alias("id"))
            .to_pandas()
        )
    path_manager = data["path_manager"]

    translator_map = {
        PathType.WALKING: dict(
            geo_data.osm_nodes.select(pl.col("rx_node_id").alias("osm"), "osm_id").rows()
        ),
        PathType.CYCLING_WALKING: dict(
            pl.concat(
                [
                    geo_data.osm_nodes.select(
                        pl.lit("W").alias("osm_id") + pl.col("osm_id").cast(pl.String)
                    ),
                    geo_data.additional_networks[NetworkType.CYCLING][0].select(
                        pl.lit("D").alias("osm_id") + pl.col("osm_id").cast(pl.String)
                    ),
                ],
                how="diagonal",
            )
            .with_row_index()
            .rows()
        ),
        PathType.DRIVING_WALKING: dict(
            pl.concat(
                [
                    geo_data.osm_nodes.select(
                        pl.lit("W").alias("osm_id") + pl.col("osm_id").cast(pl.String)
                    ),
                    geo_data.additional_networks[NetworkType.DRIVING][0].select(
                        pl.lit("D").alias("osm_id") + pl.col("osm_id").cast(pl.String)
                    ),
                ],
                how="diagonal",
            )
            .with_row_index()
            .rows()
        ),
        PathType.PUBLIC_TRANSPORT: None,
    }
    stops_by_id = pl.read_parquet(gtfs_clean_stops).to_pandas().set_index("stop_id", drop=True)

    color_map = {
        "Shops": "orange",
        "Grocery": "red",
        "Parks": "green",
        "Education": "blue",
        "Banks": "violet",
        "Health": "darkgreen",
        "Sustenance": "yellow",
    }

    plot_paths_on_map(
        labels=labels,
        nodes=nodes,
        path_manager=path_manager,
        translator_map=translator_map,
        color_map=color_map,
        path_name=output_path,
        stops_by_id=stops_by_id,
    )


if __name__ == "__main__":
    city_name = "cologne"

    with open(pathlib.Path(__file__).parent.resolve() / "config.toml", "rb") as f:
        settings = tomllib.load(f)

    setup("INFO")
    figures_directory: pathlib.Path = (
        (pathlib.Path(__file__).parent.parent.resolve() / "figures")
        / "mcr5"
        / f"{city_name}_reduced_{settings['timestamp']['timestamp']}"
    )
    figures_directory.mkdir(parents=True, exist_ok=True)
    data_directory = pathlib.Path(__file__).parent.parent.resolve() / "data"
    base_directory: pathlib.Path = data_directory / settings["timestamp"]["timestamp"]
    osm_path = base_directory / "osm_raw"
    cache_path = base_directory / "cache"
    geometa_path = base_directory / f"cache/{city_name}_geometa.json"
    gtfs_clean_dir = base_directory / f"gtfs_clean/{city_name}/"
    gtfs_clean_stops = gtfs_clean_dir / "stops.parquet"

    mcr5_output_path: pathlib.Path = base_directory / f"mcr5_results/{city_name}_reduced_paths"

    _, geo_data = mcr_py.helper_functions.load_auxiliary_classes(
        geo_meta_path=geometa_path,
        city_id=settings["city"][city_name]["german_alt"],
        osm_path=osm_path,
        cache_path=cache_path,
    )

    for mode_setting in mcr5_output_path.iterdir():
        if not mode_setting.is_dir():
            continue
        for file in mode_setting.iterdir():
            fig_path = (
                figures_directory
                / f"{file.name.replace('.pkl', '')}_{mode_setting.name}_paths.png"
            )
            if (
                file.suffix == ".pkl" and file.name != "errors.pkl"
            ):  # and not fig_path.exists():
                generate_path_image(
                    file,
                    fig_path,
                    geo_data,
                    gtfs_clean_stops,
                )
