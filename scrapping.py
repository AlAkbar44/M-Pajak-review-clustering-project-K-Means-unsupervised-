from google_play_scraper import Sort, reviews
import pandas as pd
import time

APP_ID = "id.go.pajak.djp"

# Kombinasi bahasa yang akan di-scrape. 'id' nangkep ulasan berbahasa Indonesia,
# 'en' nangkep yang bahasa Inggris (banyak juga muncul di Play Store app ini).
LANGUAGES = ["id", "en"]
COUNTRY = "id"

# Kombinasi sort — NEWEST kadang mentok lebih cepat dibanding MOST_RELEVANT,
# jadi kita coba dua-duanya lalu gabung & dedup by reviewId.
SORT_OPTIONS = [Sort.NEWEST, Sort.MOST_RELEVANT]

TARGET_PER_COMBO = 8000   # target maksimal tiap kombinasi lang+sort
BATCH_SIZE = 200


def scrape_combo(lang, sort_option, target_count):
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < target_count:
        result, continuation_token = reviews(
            APP_ID,
            lang=lang,
            country=COUNTRY,
            sort=sort_option,
            count=BATCH_SIZE,
            continuation_token=continuation_token,
        )

        if not result:
            break

        all_reviews.extend(result)

        if continuation_token is None:
            break

        time.sleep(1)  # jeda sopan biar tidak dianggap spam request

    return all_reviews[:target_count]


if __name__ == "__main__":
    print(f"Mulai scraping ulasan untuk app_id: {APP_ID}")
    all_results = []

    for lang in LANGUAGES:
        for sort_option in SORT_OPTIONS:
            sort_name = "NEWEST" if sort_option == Sort.NEWEST else "MOST_RELEVANT"
            print(f"\n>> Scraping lang='{lang}' sort={sort_name} ...")
            batch = scrape_combo(lang, sort_option, TARGET_PER_COMBO)
            print(f"   Didapat {len(batch)} ulasan dari kombinasi ini.")
            all_results.extend(batch)

    df = pd.DataFrame(all_results)
    print(f"\nTotal sebelum dedup: {len(df)} baris")

    # Dedup berdasarkan reviewId (satu ulasan bisa kepindai dari lebih dari 1 kombinasi)
    df = df.drop_duplicates(subset="reviewId").reset_index(drop=True)
    print(f"Total setelah dedup by reviewId: {len(df)} baris")

    kolom_dipakai = [
        "reviewId", "userName", "content", "score",
        "thumbsUpCount", "reviewCreatedVersion", "at", "appVersion"
    ]
    kolom_tersedia = [k for k in kolom_dipakai if k in df.columns]
    df = df[kolom_tersedia]

    output_file = "mpajak_reviews_raw.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nSelesai! Total {len(df)} ulasan unik tersimpan ke '{output_file}'")
    print(df.head(10))