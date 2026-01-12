import pandas as pd

from load_movies_dataset import load_movies_dataset, dataframe_to_vectors
from kd_tree import KDTree
from lsh import MinHashLSH
from quad_tree import QuadTree
import time
from r_tree import RTree
from range_tree import RangeTree

from similarity import jaccard_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#similarity
def get_top_n_similar(df, candidate_indices, query_genres, N=5):
    similarities = []

    for idx in candidate_indices:
        movie_genres = df.iloc[idx]["genre_names"]
        score = jaccard_similarity(query_genres, movie_genres)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
     # Keep only top-N with score > 0 for sanity
    top_n = [(idx, score) for idx, score in similarities[:N] if score > 0]

    return top_n


def top_n_similarity(df, indices, text_column, query_text, N=5):
    """
    Επιστρέφει τα N πιο όμοια movies βάσει TF-IDF cosine similarity
    σε ένα συγκεκριμένο textual attribute π.χ. genre_names.
    """

    # Extract text only for these indices
    subset_texts = df.iloc[indices][text_column].apply(lambda x: " ".join(x)).tolist()

    # Build corpus: first entry = query
    corpus = [query_text] + subset_texts

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute cosine similarity of query vs all
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Top N indices (descending)
    top_idxs = sims.argsort()[::-1][:N]
    top_scores = sims[top_idxs]

    # Map back to original movie indices
    original_indices = [indices[i] for i in top_idxs]

    results = []
    for idx, score in zip(original_indices, top_scores):
        title = df.iloc[idx]["title"]
        results.append((idx, title, score))

    return results


def run_kd_lsh(df, data_points, text_corpus):
    print("=== KD-Tree + LSH ===")

    # 1. Ορισμός 5D range bounds
    lower_bounds = [
        2000,
        3.0,
        3.0,
        30.0,
        df["budget"].quantile(0.25)
    ]
    upper_bounds = [
        2020,
        6.0,
        5.0,
        60.0,
        df["budget"].quantile(0.75)
    ]

    # 2. Build KD-tree
    start = time.perf_counter()
    items = [(vec, i) for i, vec in enumerate(data_points)]
    kd_tree = KDTree(items, dimensions=["year", "popularity", "vote_average", "runtime", "budget"])
    build_time = time.perf_counter() - start

    print(f"KD build time: {build_time:.4f}s")

    # 3. Range Query
    start = time.perf_counter()
    kd_results = kd_tree.query_range(lower_bounds, upper_bounds)
    range_time = time.perf_counter() - start

    print(f"KD range query time: {range_time:.4f}s")
    print(f"KD found {len(kd_results)} results in range.")

    # 4. LSH similarity (based on genres) — MinHashLSH
    lsh = MinHashLSH(num_perm=80, num_bands=20)

    start = time.perf_counter()
    for idx, tokens in enumerate(df["genre_names"]):
        lsh.add(tokens, idx)
    lsh_build_time = time.perf_counter() - start

    query_index = 0
    query_tokens = df.iloc[query_index]["genre_names"]

    start = time.perf_counter()
    lsh_indices = lsh.query(query_tokens)
    lsh_query_time = time.perf_counter() - start

    print(f"LSH build time: {lsh_build_time:.4f}s")
    print(f"LSH query time: {lsh_query_time:.4f}s")
    print(f"LSH returned {len(lsh_indices)} candidates")

    kd_indices = kd_results   # ήδη indices
    combined = set(kd_indices).intersection(lsh_indices)

    print("Combined results:", combined)

    # === Top-N Similarity (Jaccard) on combined results ===
    query_genres = df.iloc[0]["genre_names"]   # μπορείς να αλλάξεις το query
    top5 = get_top_n_similar(df, list(combined), query_genres, N=5)

    print("\nTop-5 similar movies based on genres (Jaccard):")
    for idx, score in top5:
        print(f"{df.iloc[idx]['title']} | score={score:.3f} | genres={df.iloc[idx]['genre_names']}")

    # Save to file
    with open("topN_results_kd.txt", "w", encoding="utf-8") as f:
        f.write("Top-5 Similar Movies (Jaccard):\n")
        for idx, score in top5:
            f.write(f"{df.iloc[idx]['title']} | score={score:.3f}\n")
    
    # 5. Save metrics
    with open("metrics_kd.txt", "w") as f:
        f.write(f"Range Execution Time: {range_time}\n")
        f.write(f"LSH Build Time: {lsh_build_time}\n")
        f.write(f"LSH Query Time: {lsh_query_time}\n")
        f.write(f"KD Build Time: {build_time}\n")
        f.write(f"Results Found: {len(combined)}\n")

   
    return range_time, lsh_build_time, lsh_query_time, len(combined)


def run_quadtree_lsh(df, data_points, text_corpus):
    print("\n=== Quad-Tree + LSH ===")

    points_2d = list(zip(df["popularity"], df["vote_average"]))
    indices = list(range(len(points_2d)))

    # Build QuadTree
    start = time.perf_counter()
    qt = QuadTree(points_2d, indices, capacity=10)
    qt_build_time = time.perf_counter() - start
    print(f"QuadTree build time: {qt_build_time:.4f}s")

    # Range Query
    x_low, x_high = 3, 6
    y_low, y_high = 3, 5

    start = time.perf_counter()
    qt_results = []
    qt.range_query(qt.root, x_low, x_high, y_low, y_high, qt_results)
    qt_range_time = time.perf_counter() - start
    print(f"QuadTree range query time: {qt_range_time:.4f}s")
    print(f"QuadTree found {len(qt_results)} results in range.")

    # LSH (MinHashLSH on genres)
    lsh = MinHashLSH(num_perm=80, num_bands=20)

    start = time.perf_counter()
    for idx, tokens in enumerate(df["genre_names"]):
        lsh.add(tokens, idx)
    lsh_build_time = time.perf_counter() - start

    query_index = 0
    query_tokens = df.iloc[query_index]["genre_names"]

    start = time.perf_counter()
    lsh_indices = lsh.query(query_tokens)
    lsh_query_time = time.perf_counter() - start

    combined = set(qt_results).intersection(lsh_indices)
    print("Combined count:", len(combined))

    # Top-N Jaccard
    query_genres = query_tokens
    top5 = get_top_n_similar(df, list(combined), query_genres, N=5)

    with open("topN_results_quadtree.txt", "w", encoding="utf-8") as f:
        f.write("Top-5 Similar Movies (Jaccard):\n")
        for idx, score in top5:
            f.write(f"{df.iloc[idx]['title']} | score={score:.3f}\n")

    with open("metrics_quadtree.txt", "w") as f:
        f.write(f"Build Time: {qt_build_time}\n")
        f.write(f"Range Execution Time: {qt_range_time}\n")
        f.write(f"LSH Build Time: {lsh_build_time}\n")
        f.write(f"LSH Query Time: {lsh_query_time}\n")
        f.write(f"Results Found: {len(combined)}\n")

    print(f"LSH build time: {lsh_build_time:.4f}s")
    print(f"LSH query time: {lsh_query_time:.4f}s")
    print(f"LSH returned {len(lsh_indices)} candidates")


    return qt_build_time, qt_range_time, lsh_build_time, lsh_query_time, len(combined)




def run_range_lsh(df, data_points, text_corpus):
    print("\n=== Range-Tree + LSH ===")

    points_2d = list(zip(df["popularity"], df["vote_average"]))
    indices = list(range(len(points_2d)))

    # BUILD RANGE-TREE
    start = time.perf_counter()
    rt = RangeTree(points_2d, indices)
    rt_build_time = time.perf_counter() - start
    print(f"RangeTree build time: {rt_build_time:.4f}s")

    # RANGE QUERY
    x_low, x_high = 3, 6
    y_low, y_high = 3, 5

    start = time.perf_counter()
    rt_results = []
    rt.range_query(rt.root, x_low, x_high, y_low, y_high, rt_results)
    rt_range_time = time.perf_counter() - start
    print(f"RangeTree range time: {rt_range_time:.4f}s")
    print(f"RangeTree returned {len(rt_results)} results.")

    # LSH (MinHashLSH on genres)
    lsh = MinHashLSH(num_perm=80, num_bands=20)

    start = time.perf_counter()
    for idx, tokens in enumerate(df["genre_names"]):
        lsh.add(tokens, idx)
    lsh_build_time = time.perf_counter() - start

    query_index = 0
    query_tokens = df.iloc[query_index]["genre_names"]

    start = time.perf_counter()
    lsh_indices = lsh.query(query_tokens)
    lsh_query_time = time.perf_counter() - start

    combined = set(rt_results).intersection(lsh_indices)
    print("Combined count:", len(combined))

    # Top-N Jaccard
    top5 = get_top_n_similar(df, list(combined), query_tokens, N=5)

    with open("topN_results_rangetree.txt", "w", encoding="utf-8") as f:
        f.write("Top-5 Similar Movies (Jaccard):\n")
        for idx, score in top5:
            f.write(f"{df.iloc[idx]['title']} | score={score:.3f}\n")

    with open("metrics_range.txt", "w") as f:
        f.write(f"Build Time: {rt_build_time}\n")
        f.write(f"Range Execution Time: {rt_range_time}\n")
        f.write(f"LSH Build Time: {lsh_build_time}\n")
        f.write(f"LSH Query Time: {lsh_query_time}\n")
        f.write(f"Results Found: {len(combined)}\n")

    return rt_build_time, rt_range_time, lsh_build_time, lsh_query_time, len(combined)



def run_rtree_lsh(df, data_points, text_corpus):
    print("\n=== R-Tree (5D) + LSH ===")

    # 5D bounds (ίδια λογική με KD)
    lower_bounds = [2000, 3.0, 3.0, 30.0, df["budget"].quantile(0.25)]
    upper_bounds = [2020, 6.0, 5.0, 60.0, df["budget"].quantile(0.75)]

    vectors_5d = data_points
    indices = list(range(len(vectors_5d)))

    # Build RTree (5D)
    start = time.perf_counter()
    rt = RTree(vectors_5d, indices)
    rt_build_time = time.perf_counter() - start
    print(f"R-Tree build time: {rt_build_time:.4f}s")

    # Range query (5D)
    start = time.perf_counter()
    rtree_results = rt.range_query_5d(lower_bounds, upper_bounds)
    rtree_range_time = time.perf_counter() - start
    print(f"R-Tree range query time: {rtree_range_time:.4f}s")
    print(f"R-Tree returned {len(rtree_results)} results.")

    # LSH (MinHashLSH on genres)
    lsh = MinHashLSH(num_perm=80, num_bands=20)

    start = time.perf_counter()
    for idx, tokens in enumerate(df["genre_names"]):
        lsh.add(tokens, idx)
    lsh_build_time = time.perf_counter() - start

    query_index = 0
    query_tokens = df.iloc[query_index]["genre_names"]

    start = time.perf_counter()
    lsh_indices = lsh.query(query_tokens)
    lsh_query_time = time.perf_counter() - start

    combined = set(rtree_results).intersection(lsh_indices)
    print("Combined count:", len(combined))

    # Top-N Jaccard
    top5 = get_top_n_similar(df, list(combined), query_tokens, N=5)

    with open("topN_results_rtree.txt", "w", encoding="utf-8") as f:
        f.write("Top-5 Similar Movies (Jaccard):\n")
        for idx, score in top5:
            f.write(f"{df.iloc[idx]['title']} | score={score:.3f}\n")

    with open("metrics_rtree.txt", "w") as f:
        f.write(f"Build Time: {rt_build_time}\n")
        f.write(f"Range Execution Time: {rtree_range_time}\n")
        f.write(f"LSH Build Time: {lsh_build_time}\n")
        f.write(f"LSH Query Time: {lsh_query_time}\n")
        f.write(f"Results Found: {len(combined)}\n")


        print(f"LSH build time: {lsh_build_time:.4f}s")
        print(f"LSH query time: {lsh_query_time:.4f}s")
        print(f"LSH returned {len(lsh_indices)} candidates")


    return rt_build_time, rtree_range_time, lsh_build_time, lsh_query_time, len(combined)

if __name__ == "__main__":
    xlsx_path = "data_movies_clean.xlsx"
    df = load_movies_dataset(xlsx_path)

    ids, titles, data_points = dataframe_to_vectors(df)

    text_corpus = df["genre_names"].apply(lambda lst: " ".join(lst)).tolist()

    run_kd_lsh(df, data_points, text_corpus)
    run_quadtree_lsh(df, data_points, text_corpus)
    run_range_lsh(df, data_points, text_corpus)
    run_rtree_lsh(df, data_points, text_corpus)

    print("DF rows:", len(df))
    print("Vectors:", len(data_points))
    print("Vector dim:", len(data_points[0]))
    print("Sample vector:", data_points[0])
