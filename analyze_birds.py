#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze bird feeder detections stored in Postgres.

The script reads the same .env / Postgres settings used by birdwatch.py,
joins detections to hourly Mansfield, MA weather from Open-Meteo, derives
sunrise/sunset-relative fields, and writes charts plus CSV summaries.
"""

import argparse
import json
import math
from bisect import bisect_right
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
from astral import LocationInfo
from astral.sun import sun
from sqlalchemy import URL, bindparam, create_engine, text

from bd.db import PostgresConfig
from bd.env import load_dotenv
from bd.wildlife_db import _quote_ident, wildlife_table_from_env


DEFAULT_LATITUDE = 42.0334
DEFAULT_LONGITUDE = -71.2189
DEFAULT_TIMEZONE = "America/New_York"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_EXCLUDED_SPECIES = (
    "no_bird",
    "mouse",
    "red_squirrel",
    "eastern_gray_squirrel",
)
WEATHER_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
    "pressure_msl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze wildlife detections from Postgres and write bird feeder charts."
    )
    parser.add_argument("--days", type=int, default=365, help="Look back this many days.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum classifier confidence to include.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=8,
        help="Limit multi-species charts to the top N species by count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for charts, CSVs, and cached weather.",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Wildlife table name. Defaults to WILDLIFE_TABLE or wildlife.",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=DEFAULT_LATITUDE,
        help="Latitude for weather and sunrise/sunset.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=DEFAULT_LONGITUDE,
        help="Longitude for weather and sunrise/sunset.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="Local timezone for Mansfield-style analysis.",
    )
    parser.add_argument(
        "--refresh-weather",
        action="store_true",
        help="Ignore cached weather and fetch Open-Meteo again.",
    )
    parser.add_argument(
        "--visit-gap-minutes",
        type=int,
        default=2,
        help="Same-species detections closer than this are one visit.",
    )
    parser.add_argument(
        "--interaction-windows",
        type=int,
        nargs="+",
        default=[5, 15],
        help="Minute windows for follow-on species interaction summaries.",
    )
    parser.add_argument(
        "--exclude-species",
        nargs="+",
        default=list(DEFAULT_EXCLUDED_SPECIES),
        help="Classifier labels to exclude from all analysis outputs.",
    )
    return parser.parse_args()


def load_detections(
    *,
    table: str,
    days: int,
    min_confidence: float,
    excluded_species: list[str],
) -> pd.DataFrame:
    t = _quote_ident(table)
    since = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=days)
    excluded_species = [s for s in excluded_species if s]
    exclusion_sql = ""
    params: dict[str, object] = {
        "since": since,
        "min_confidence": min_confidence,
    }
    if excluded_species:
        exclusion_sql = "          AND classifier_label NOT IN :excluded_species\n"
        params["excluded_species"] = excluded_species

    sql = text(
        f"""
        SELECT
            id,
            detected_at,
            image_path,
            detector_label,
            detector_confidence,
            classifier_label,
            classifier_confidence,
            video_source
        FROM {t}
        WHERE detected_at >= :since
          AND classifier_label IS NOT NULL
          AND classifier_confidence >= :min_confidence
{exclusion_sql.rstrip()}
        ORDER BY detected_at;
    """
    )
    if excluded_species:
        sql = sql.bindparams(bindparam("excluded_species", expanding=True))

    engine = create_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return df

    df["detected_at_utc"] = pd.to_datetime(df["detected_at"], utc=True)
    df["species"] = df["classifier_label"].astype(str)
    return df


def create_db_engine():
    cfg = PostgresConfig.from_env()
    if cfg.dsn:
        return create_engine(sqlalchemy_dsn(cfg.dsn))

    query: dict[str, str] = {}
    if cfg.sslmode:
        query["sslmode"] = cfg.sslmode
    if cfg.connect_timeout_s is not None:
        query["connect_timeout"] = str(cfg.connect_timeout_s)
    if cfg.application_name:
        query["application_name"] = cfg.application_name

    url = URL.create(
        "postgresql+psycopg",
        username=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        database=cfg.dbname,
        query=query,
    )
    return create_engine(url)


def sqlalchemy_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://"):]
    if dsn.startswith("postgres://"):
        return "postgresql+psycopg://" + dsn[len("postgres://"):]
    return dsn


def detection_date_range(df: pd.DataFrame, tz: ZoneInfo) -> tuple[date, date]:
    local = df["detected_at_utc"].dt.tz_convert(tz)
    return local.min().date(), local.max().date()


def fetch_weather(
    *,
    start_date: date,
    end_date: date,
    latitude: float,
    longitude: float,
    timezone_name: str,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (
        f"open_meteo_{latitude:.4f}_{longitude:.4f}_"
        f"{start_date.isoformat()}_{end_date.isoformat()}_hourly.csv"
    )
    cache_path = cache_dir / key

    if cache_path.exists() and not refresh:
        weather = pd.read_csv(cache_path)
    else:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": timezone_name,
        }
        response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload or "time" not in payload["hourly"]:
            raise RuntimeError(f"Unexpected Open-Meteo response: {json.dumps(payload)[:500]}")
        weather = pd.DataFrame(payload["hourly"])
        weather.to_csv(cache_path, index=False)

    weather["local_hour"] = pd.to_datetime(weather["time"])
    return weather


def add_time_fields(df: pd.DataFrame, *, tz: ZoneInfo) -> pd.DataFrame:
    out = df.copy()
    out["detected_at_local"] = out["detected_at_utc"].dt.tz_convert(tz)
    out["local_hour"] = out["detected_at_local"].dt.floor("h").dt.tz_localize(None)
    out["hour"] = out["detected_at_local"].dt.hour
    out["local_date"] = out["detected_at_local"].dt.date
    local_naive = out["detected_at_local"].dt.tz_localize(None)
    out["month"] = local_naive.dt.to_period("M").dt.to_timestamp()
    return out


def add_sun_fields(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.DataFrame:
    out = df.copy()
    tz = ZoneInfo(timezone_name)
    location = LocationInfo("Mansfield", "USA", timezone_name, latitude, longitude)
    dates: Iterable[date] = sorted(out["local_date"].dropna().unique())
    sun_by_date = {
        d: sun(location.observer, date=d, tzinfo=tz)
        for d in dates
    }

    def minutes_since_sunrise(row: pd.Series) -> float:
        return (row["detected_at_local"] - sun_by_date[row["local_date"]]["sunrise"]).total_seconds() / 60.0

    def minutes_until_sunset(row: pd.Series) -> float:
        return (sun_by_date[row["local_date"]]["sunset"] - row["detected_at_local"]).total_seconds() / 60.0

    out["minutes_since_sunrise"] = out.apply(minutes_since_sunrise, axis=1)
    out["minutes_until_sunset"] = out.apply(minutes_until_sunset, axis=1)
    out["sunrise_relative_hour"] = (out["minutes_since_sunrise"] // 60).astype(int)
    out["sunset_relative_hour"] = ((-out["minutes_until_sunset"]) // 60).astype(int)
    out["daylight_phase"] = out.apply(classify_daylight_phase, axis=1)
    return out


def classify_daylight_phase(row: pd.Series) -> str:
    since = row["minutes_since_sunrise"]
    until = row["minutes_until_sunset"]
    if since < -60:
        return "night_before_dawn"
    if since < 0:
        return "pre_dawn"
    if since < 180:
        return "morning"
    if until > 180:
        return "midday"
    if until > 0:
        return "afternoon_dusk"
    return "night_after_sunset"


def join_weather(detections: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return detections.merge(weather, on="local_hour", how="left")


def top_species(df: pd.DataFrame, limit: int) -> list[str]:
    return df["species"].value_counts().head(limit).index.tolist()


def save_counts_csv(df: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        df.groupby("species", as_index=False)
        .agg(
            detections=("id", "count"),
            first_seen=("detected_at_local", "min"),
            last_seen=("detected_at_local", "max"),
            mean_confidence=("classifier_confidence", "mean"),
        )
        .sort_values("detections", ascending=False)
    )
    counts.to_csv(output_dir / "species_summary.csv", index=False)


def plot_species_by_hour(df: pd.DataFrame, species: list[str], output_dir: Path) -> None:
    data = df[df["species"].isin(species)]
    pivot = (
        data.groupby(["hour", "species"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
    )
    ax = pivot.plot(marker="o", figsize=(12, 7))
    add_average_line(ax, pivot)
    ax.set_title("Bird detections by clock hour")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Detections")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "species_by_hour.png", dpi=150)
    plt.close()
    pivot.to_csv(output_dir / "species_by_hour.csv")


def plot_species_by_sunrise(df: pd.DataFrame, species: list[str], output_dir: Path) -> None:
    data = df[df["species"].isin(species)].copy()
    if data.empty:
        return
    min_bin = max(-6, int(data["sunrise_relative_hour"].min()))
    max_bin = min(18, int(data["sunrise_relative_hour"].max()))
    bins = range(min_bin, max_bin + 1)
    pivot = (
        data.groupby(["sunrise_relative_hour", "species"])
        .size()
        .unstack(fill_value=0)
        .reindex(bins, fill_value=0)
    )
    ax = pivot.plot(marker="o", figsize=(12, 7))
    add_average_line(ax, pivot)
    ax.set_title("Bird detections relative to sunrise")
    ax.set_xlabel("Hours since sunrise")
    ax.set_ylabel("Detections")
    ax.set_xticks(list(bins))
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "species_by_sunrise_relative.png", dpi=150)
    plt.close()
    pivot.to_csv(output_dir / "species_by_sunrise_relative.csv")


def plot_species_by_sunset(df: pd.DataFrame, species: list[str], output_dir: Path) -> None:
    data = df[df["species"].isin(species)].copy()
    if data.empty:
        return
    min_bin = max(-18, int(data["sunset_relative_hour"].min()))
    max_bin = min(6, int(data["sunset_relative_hour"].max()))
    bins = range(min_bin, max_bin + 1)
    pivot = (
        data.groupby(["sunset_relative_hour", "species"])
        .size()
        .unstack(fill_value=0)
        .reindex(bins, fill_value=0)
    )
    ax = pivot.plot(marker="o", figsize=(12, 7))
    add_average_line(ax, pivot)
    ax.set_title("Bird detections relative to sunset")
    ax.set_xlabel("Hours relative to sunset (negative = before sunset)")
    ax.set_ylabel("Detections")
    ax.set_xticks(list(bins))
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "species_by_sunset_relative.png", dpi=150)
    plt.close()
    pivot.to_csv(output_dir / "species_by_sunset_relative.csv")


def plot_species_by_month(df: pd.DataFrame, species: list[str], output_dir: Path) -> None:
    data = df[df["species"].isin(species)]
    if data.empty:
        return
    month_index = pd.date_range(data["month"].min(), data["month"].max(), freq="MS")
    pivot = (
        data.groupby(["month", "species"])
        .size()
        .unstack(fill_value=0)
        .reindex(month_index, fill_value=0)
    )
    ax = pivot.plot(marker="o", figsize=(12, 7))
    add_average_line(ax, pivot)
    ax.set_title("Monthly detections by species")
    ax.set_xlabel("Month")
    ax.set_ylabel("Detections")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "species_by_month.png", dpi=150)
    plt.close()
    pivot.to_csv(output_dir / "species_by_month.csv")


def add_average_line(ax, pivot: pd.DataFrame) -> None:
    if pivot.empty:
        return
    average = pivot.mean(axis=1).rename("Average")
    average.plot(
        ax=ax,
        color="black",
        linewidth=4,
        zorder=10,
    )


def build_hourly_activity(detections: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    counts = detections.groupby("local_hour").size().rename("detections").reset_index()
    hourly = weather.merge(counts, on="local_hour", how="left")
    hourly["detections"] = hourly["detections"].fillna(0).astype(int)
    return hourly


def weather_bin_summary(
    hourly: pd.DataFrame,
    *,
    column: str,
    bins: int | list[float],
    label: str,
) -> pd.DataFrame:
    rows = hourly[["detections", column]].dropna().copy()
    if rows.empty:
        return pd.DataFrame()
    rows["bin"] = pd.cut(rows[column], bins=bins, duplicates="drop", include_lowest=True)
    summary = (
        rows.groupby("bin", observed=True)
        .agg(
            hours=("detections", "size"),
            total_detections=("detections", "sum"),
            mean_detections_per_hour=("detections", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "weather_variable", label)
    summary["bin"] = summary["bin"].astype(str)
    return summary


def plot_weather_summary(hourly: pd.DataFrame, output_dir: Path) -> None:
    summaries = [
        weather_bin_summary(
            hourly,
            column="temperature_2m",
            bins=8,
            label="temperature_2m",
        ),
        weather_bin_summary(
            hourly,
            column="cloud_cover",
            bins=[0, 20, 40, 60, 80, 100],
            label="cloud_cover",
        ),
        weather_bin_summary(
            hourly,
            column="precipitation",
            bins=[-0.001, 0, 0.1, 1, 5, math.inf],
            label="precipitation",
        ),
        weather_bin_summary(
            hourly,
            column="wind_speed_10m",
            bins=8,
            label="wind_speed_10m",
        ),
    ]
    summaries = [s for s in summaries if not s.empty]
    if not summaries:
        hourly.to_csv(output_dir / "hourly_weather_activity.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "weather_summary.csv", index=False)
        return

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output_dir / "weather_summary.csv", index=False)
    hourly.to_csv(output_dir / "hourly_weather_activity.csv", index=False)

    for variable, group in summary.groupby("weather_variable"):
        group = group.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(group)))
        ax.bar(x, group["mean_detections_per_hour"])
        ax.set_title(f"Mean detections per hour by {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Mean detections per hour")
        ax.set_xticks(x)
        ax.set_xticklabels(group["bin"], rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / f"detections_vs_{variable}.png", dpi=150)
        plt.close(fig)


def collapse_visits(df: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    visits: list[dict[str, object]] = []
    gap = pd.Timedelta(minutes=gap_minutes)
    for species, group in df.sort_values("detected_at_utc").groupby("species", sort=False):
        current_start = None
        current_end = None
        count = 0
        for row in group.itertuples(index=False):
            ts = row.detected_at_utc
            if current_start is None or ts - current_end > gap:
                if current_start is not None:
                    visits.append(
                        {
                            "species": species,
                            "visit_start_utc": current_start,
                            "visit_end_utc": current_end,
                            "detections": count,
                        }
                    )
                current_start = ts
                count = 1
            else:
                count += 1
            current_end = ts
        if current_start is not None:
            visits.append(
                {
                    "species": species,
                    "visit_start_utc": current_start,
                    "visit_end_utc": current_end,
                    "detections": count,
                }
            )
    return pd.DataFrame(visits).sort_values("visit_start_utc").reset_index(drop=True)


def compute_interactions(visits: pd.DataFrame, windows: list[int], output_dir: Path) -> None:
    if visits.empty:
        pd.DataFrame().to_csv(output_dir / "species_interactions.csv", index=False)
        return

    species_list = sorted(visits["species"].unique())
    starts_by_species = {
        sp: sorted(visits.loc[visits["species"] == sp, "visit_start_utc"].tolist())
        for sp in species_list
    }
    total_visits = len(visits)
    baseline_share = {
        sp: len(starts) / total_visits
        for sp, starts in starts_by_species.items()
    }
    rows: list[dict[str, object]] = []

    for window in windows:
        delta = pd.Timedelta(minutes=window)
        for trigger in species_list:
            trigger_starts = starts_by_species[trigger]
            for following in species_list:
                if following == trigger:
                    continue
                follow_starts = starts_by_species[following]
                observed = 0
                for start in trigger_starts:
                    right_after = bisect_right(follow_starts, start)
                    within_window = bisect_right(follow_starts, start + delta)
                    if within_window > right_after:
                        observed += 1
                observed_rate = observed / len(trigger_starts) if trigger_starts else 0.0
                baseline = baseline_share[following]
                rows.append(
                    {
                        "trigger_species": trigger,
                        "following_species": following,
                        "window_minutes": window,
                        "trigger_visits": len(trigger_starts),
                        "observed_followups": observed,
                        "observed_rate": observed_rate,
                        "baseline_visit_share": baseline,
                        "lift_vs_baseline": observed_rate / baseline if baseline else None,
                    }
                )

    interactions = pd.DataFrame(rows).sort_values(
        ["window_minutes", "lift_vs_baseline", "observed_followups"],
        ascending=[True, False, False],
    )
    visits.to_csv(output_dir / "species_visits.csv", index=False)
    interactions.to_csv(output_dir / "species_interactions.csv", index=False)
    plot_interaction_heatmaps(interactions, output_dir)


def plot_interaction_heatmaps(interactions: pd.DataFrame, output_dir: Path) -> None:
    if interactions.empty:
        return

    for window, group in interactions.groupby("window_minutes"):
        species = sorted(
            set(group["trigger_species"].dropna().astype(str))
            | set(group["following_species"].dropna().astype(str))
        )
        if not species:
            continue

        lift = (
            group.pivot(
                index="trigger_species",
                columns="following_species",
                values="lift_vs_baseline",
            )
            .reindex(index=species, columns=species)
            .fillna(0.0)
        )
        rate = (
            group.pivot(
                index="trigger_species",
                columns="following_species",
                values="observed_rate",
            )
            .reindex(index=species, columns=species)
            .fillna(0.0)
        )

        plot_heatmap(
            lift,
            title=f"Interaction lift within {window} minutes",
            output_path=output_dir / f"species_interaction_lift_{window}m.png",
            colorbar_label="Observed / baseline",
        )
        plot_heatmap(
            rate,
            title=f"Follow-on probability within {window} minutes",
            output_path=output_dir / f"species_interaction_rate_{window}m.png",
            colorbar_label="Share of trigger visits",
        )


def plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    colorbar_label: str,
) -> None:
    width = max(8, 0.55 * len(matrix.columns) + 4)
    height = max(7, 0.55 * len(matrix.index) + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Following species")
    ax.set_ylabel("Trigger species")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env", override=False)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table = args.table or wildlife_table_from_env()
    tz = ZoneInfo(args.timezone)

    detections = load_detections(
        table=table,
        days=args.days,
        min_confidence=args.min_confidence,
        excluded_species=args.exclude_species,
    )
    if detections.empty:
        print("No detections matched the requested filters.")
        return

    detections = add_time_fields(detections, tz=tz)
    start_date, end_date = detection_date_range(detections, tz)

    weather = fetch_weather(
        start_date=start_date,
        end_date=end_date,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone_name=args.timezone,
        cache_dir=output_dir / "cache",
        refresh=args.refresh_weather,
    )
    detections = join_weather(detections, weather)
    detections = add_sun_fields(
        detections,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone_name=args.timezone,
    )

    species = top_species(detections, args.top_species)
    detections.to_csv(output_dir / "detections_enriched.csv", index=False)
    save_counts_csv(detections, output_dir)
    plot_species_by_hour(detections, species, output_dir)
    plot_species_by_sunrise(detections, species, output_dir)
    plot_species_by_sunset(detections, species, output_dir)
    plot_species_by_month(detections, species, output_dir)

    hourly = build_hourly_activity(detections, weather)
    plot_weather_summary(hourly, output_dir)

    visits = collapse_visits(detections, args.visit_gap_minutes)
    compute_interactions(visits, args.interaction_windows, output_dir)

    print(f"Analyzed {len(detections)} detections across {len(species)} plotted species.")
    print(f"Wrote charts and CSV summaries to: {output_dir}")


if __name__ == "__main__":
    main()
