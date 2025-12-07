import argparse
import sqlite3

import pandas as pd


def analyze_parking_occupancy(db_path: str, interval_minutes: int = 15):
    """
    Analyze parking occupancy over time using the State table in SQLite.
    """

    conn = sqlite3.connect(db_path)

    # Select from the new 'state' table
    query = """
    SELECT
        timestamp,
        parked_count,
        moving_count,
        total_count
    FROM state
    ORDER BY timestamp ASC
    """

    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if df.empty:
        print("No data found in database.")
        return

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Round timestamps to the nearest interval
    df["time_bucket"] = df["timestamp"].dt.floor(f"{interval_minutes}min")

    # Since we have per-frame data, we should probably take the MAX or MEAN occupancy within the interval
    # Taking the MAX gives us the peak occupancy in that 15 min slot.

    print(f"\n--- Parking Occupancy (Peak) per {interval_minutes} min ---")

    occupancy = df.groupby("time_bucket")[
        ["parked_count", "moving_count", "total_count"]
    ].max()

    print(occupancy)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Parking DB")
    parser.add_argument("--db", default="data/parking.db", help="Path to database")
    parser.add_argument(
        "--interval", type=int, default=15, help="Time interval in minutes"
    )

    args = parser.parse_args()

    analyze_parking_occupancy(args.db, args.interval)
