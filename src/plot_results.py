import os
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Convert seconds → MM:SS
# -------------------------
def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# -------------------------
# Plot function
# -------------------------
def plot_file(csv_path):
    df = pd.read_csv(csv_path)

    time = df["time_sec"]
    score = df["audio_score"]

    plt.figure(figsize=(14, 6))
    plt.plot(time, score, label="Audio Score")

    # Highlight peaks
    threshold = 0.6
    peaks = df[df["audio_score"] > threshold]

    plt.scatter(peaks["time_sec"], peaks["audio_score"],
                label="Peaks", marker="o")

    # 🔥 Improve x-axis readability
    tick_positions = time[::len(time)//10]  # ~10 ticks
    tick_labels = [format_time(t) for t in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(f"Audio Excitement — {os.path.basename(csv_path)}")
    plt.xlabel("Match Time (MM:SS)")
    plt.ylabel("Excitement Score (0–1)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 🔥 Print top moments in readable format
    print("\nTop excitement moments:")
    top = df.nlargest(5, "audio_score")
    for _, row in top.iterrows():
        print(f"{format_time(row['time_sec'])} → score={row['audio_score']:.2f}")


# -------------------------
# Main
# -------------------------
def main():
    input_dir = "data/audio_scores"

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            csv_path = os.path.join(input_dir, file)
            print(f"Plotting: {file}")
            plot_file(csv_path)


if __name__ == "__main__":
    main()
