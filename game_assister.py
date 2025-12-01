import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def build_category_lists(games_df: pd.DataFrame, users_df: pd.DataFrame):
    # 장르 목록 (게임 + 유저 선호 장르에서 모두 수집)
    genre_set = set(games_df["genre"].dropna().unique().tolist())
    for gpref in users_df["preferred_genres"].dropna():
        for g in str(gpref).split(";"):
            g = g.strip()
            if g:
                genre_set.add(g)
    genres = sorted(genre_set)

    # 스토리 목록 (게임 + 유저 선호 스토리에서 모두 수집)
    story_set = set()
    for s in games_df["stories"].dropna():
        for t in str(s).split(";"):
            t = t.strip()
            if t:
                story_set.add(t)
    for spref in users_df["preferred_stories"].dropna():
        for t in str(spref).split(";"):
            t = t.strip()
            if t:
                story_set.add(t)
    stories = sorted(story_set)

    # 나이 버킷 (예: "10대", "20대"...)
    age_buckets = sorted(users_df["age_group"].dropna().unique().tolist())

    # 성별 (예: "남", "여")
    genders = sorted(users_df["gender"].dropna().unique().tolist())

    return age_buckets, genders, genres, stories


class GameRecommenderFromFile:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.games_df = pd.read_excel(excel_path, sheet_name="games")
        self.users_df = pd.read_excel(excel_path, sheet_name="users")

        (
            self.AGE_BUCKETS,
            self.GENDERS,
            self.GENRES,
            self.STORIES,
        ) = build_category_lists(self.games_df, self.users_df)

        # 카테고리 → 인덱스 매핑
        self.age2idx = {a: i for i, a in enumerate(self.AGE_BUCKETS)}
        self.gender2idx = {g: i for i, g in enumerate(self.GENDERS)}
        self.genre2idx = {g: i for i, g in enumerate(self.GENRES)}
        self.story2idx = {s: i for i, s in enumerate(self.STORIES)}

        self.SIM_DIM = (
            len(self.AGE_BUCKETS)
            + len(self.GENDERS)
            + len(self.GENRES)
            + len(self.STORIES)
        )

        # 게임 벡터 & 인기 점수 미리 계산
        self.game_vectors: Dict[str, np.ndarray] = {}
        self.popularity: Dict[str, float] = {}
        self._build_games()

    # ---------------- 인코딩 유틸 ----------------

    def _encode_age_bucket_from_group(self, age_group: str) -> np.ndarray:
        vec = np.zeros(len(self.AGE_BUCKETS), dtype=float)
        if age_group in self.age2idx:
            vec[self.age2idx[age_group]] = 1.0
        return vec

    def _encode_gender(self, gender: str) -> np.ndarray:
        vec = np.zeros(len(self.GENDERS), dtype=float)
        if gender in self.gender2idx:
            vec[self.gender2idx[gender]] = 1.0
        return vec

    def _encode_gender_ratio_from_trend(self, trend: str) -> np.ndarray:
        """
        games.gender_trend (Male-dominant / Female-dominant / Balanced)
        를 대략적인 남성비율로 매핑
        """
        trend = str(trend)
        if trend == "Male-dominant":
            male_ratio = 0.8
        elif trend == "Female-dominant":
            male_ratio = 0.2
        elif trend == "Balanced":
            male_ratio = 0.5
        else:
            male_ratio = 0.5
        male = np.clip(male_ratio, 0.0, 1.0)
        female = 1.0 - male
        return np.array([male, female], dtype=float)

    def _encode_multi_hot(
        self,
        items_str: str,
        mapping: Dict[str, int],
        size: int,
        normalize=True,
    ) -> np.ndarray:
        vec = np.zeros(size, dtype=float)
        if pd.isna(items_str):
            return vec
        for item in str(items_str).split(";"):
            item = item.strip()
            if item and item in mapping:
                vec[mapping[item]] = 1.0
        if normalize and vec.sum() > 0:
            vec = vec / vec.sum()
        return vec

    def _encode_single_hot(self, item: str, mapping: Dict[str, int], size: int) -> np.ndarray:
        vec = np.zeros(size, dtype=float)
        if pd.isna(item):
            return vec
        item = str(item).strip()
        if item in mapping:
            vec[mapping[item]] = 1.0
        return vec

    # ---------------- 유저/게임 벡터 ----------------

    def _encode_user_for_similarity(self, user_row: pd.Series) -> np.ndarray:
        """
        users 시트 한 행을 받아서 유저 벡터로 변환
        """
        age_vec = self._encode_age_bucket_from_group(user_row["age_group"])
        gender_vec = self._encode_gender(user_row["gender"])

        genre_vec = self._encode_multi_hot(
            user_row.get("preferred_genres", ""),
            self.genre2idx,
            len(self.GENRES),
            normalize=True,
        )

        story_vec = self._encode_multi_hot(
            user_row.get("preferred_stories", ""),
            self.story2idx,
            len(self.STORIES),
            normalize=True,
        )

        return np.concatenate([age_vec, gender_vec, genre_vec, story_vec])

    def _encode_game_for_similarity(self, game_row: pd.Series) -> np.ndarray:
        """
        games 시트 한 행을 받아서 게임 벡터로 변환
        """
        # 평균 이용자 나이 → 나이대 버킷으로 근사
        avg_age = game_row.get("avg_user_age", np.nan)
        if pd.isna(avg_age):
            # 값 없으면 중간 나이대로
            mid_idx = len(self.AGE_BUCKETS) // 2
            age_vec = np.zeros(len(self.AGE_BUCKETS), dtype=float)
            age_vec[mid_idx] = 1.0
        else:
            try:
                age_val = float(avg_age)
            except Exception:
                age_val = 25.0

            # 대강 10대/20대/... 나누기 (엑셀에 맞게 수정해도 됨)
            if age_val < 20:
                age_group = "10대"
            elif age_val < 30:
                age_group = "20대"
            elif age_val < 40:
                age_group = "30대"
            elif age_val < 50:
                age_group = "40대"
            else:
                age_group = "50대"

            if age_group not in self.age2idx:
                age_vec = np.zeros(len(self.AGE_BUCKETS), dtype=float)
                mid_idx = len(self.AGE_BUCKETS) // 2
                age_vec[mid_idx] = 1.0
            else:
                age_vec = self._encode_age_bucket_from_group(age_group)

        gender_vec = self._encode_gender_ratio_from_trend(
            game_row.get("gender_trend", "")
        )

        genre_vec = self._encode_single_hot(
            game_row.get("genre", ""),
            self.genre2idx,
            len(self.GENRES),
        )

        story_vec = self._encode_multi_hot(
            game_row.get("stories", ""),
            self.story2idx,
            len(self.STORIES),
            normalize=False,
        )

        return np.concatenate([age_vec, gender_vec, genre_vec, story_vec])

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---------------- 게임 테이블 빌드 ----------------

    def _build_games(self):
        # 게임 벡터 미리 계산
        for _, row in self.games_df.iterrows():
            gid = row["game_id"]
            self.game_vectors[gid] = self._encode_game_for_similarity(row)

        # 인기 점수 (avg_user_count 정규화)
        counts = self.games_df["avg_user_count"].fillna(0).to_numpy(dtype=float)
        if len(counts) > 0:
            min_c, max_c = counts.min(), counts.max()
            if max_c == min_c:
                norm_counts = np.ones_like(counts)
            else:
                norm_counts = (counts - min_c) / (max_c - min_c)
        else:
            norm_counts = np.array([])

        for (idx, row), c in zip(self.games_df.iterrows(), norm_counts):
            gid = row["game_id"]
            self.popularity[gid] = float(c)

    # ---------------- 단일 유저 추천 ----------------

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        w_user: float = 0.6,
        w_last: float = 0.2,
        w_pop: float = 0.2,
    ):
        """
        주어진 user_id에 대해 상위 top_k개 게임을 추천.
        users 시트의 game_id는 '마지막으로 플레이한 게임'이라고 가정.
        """
        user_rows = self.users_df[self.users_df["user_id"] == user_id]
        if user_rows.empty:
            raise ValueError(f"user_id {user_id} not found")

        user_row = user_rows.iloc[0]
        user_vec = self._encode_user_for_similarity(user_row)

        # 마지막 플레이 게임
        last_game_id = user_row.get("game_id", None)
        last_vec = None
        if isinstance(last_game_id, str) and last_game_id in self.game_vectors:
            last_vec = self.game_vectors[last_game_id]
        else:
            # 없으면 w_last를 w_user로 합치기
            w_user = w_user + w_last
            w_last = 0.0

        scores = []
        for _, g_row in self.games_df.iterrows():
            gid = g_row["game_id"]
            gvec = self.game_vectors[gid]

            sim_user = self._cosine_similarity(user_vec, gvec)
            sim_last = 0.0
            if last_vec is not None:
                sim_last = self._cosine_similarity(last_vec, gvec)
            pop = self.popularity.get(gid, 0.0)

            score = w_user * sim_user + w_last * sim_last + w_pop * pop
            scores.append((score, g_row))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for s, row in scores[:top_k]:
            info = row.to_dict()
            info["score"] = s
            results.append(info)
        return results

    # ---------------- ★ 전체 유저 추천 엑셀 출력 ----------------

    def export_all_recommendations(
        self,
        output_path: str,
        top_k: int = 5,
        w_user: float = 0.6,
        w_last: float = 0.2,
        w_pop: float = 0.2,
    ):
        """
        모든 유저에 대해 상위 top_k개 추천 게임을 계산하고
        결과를 엑셀 파일로 저장.
        output_path: 예) "user_game_recommendations.xlsx"
        """
        records = []

        user_ids = self.users_df["user_id"].unique().tolist()

        for uid in user_ids:
            try:
                recs = self.recommend_for_user(
                    uid,
                    top_k=top_k,
                    w_user=w_user,
                    w_last=w_last,
                    w_pop=w_pop,
                )
            except Exception as e:
                print(f"[WARN] user {uid} 추천 중 오류 발생: {e}")
                continue

            for rank, r in enumerate(recs, start=1):
                records.append(
                    {
                        "user_id": uid,
                        "rank": rank,
                        "game_id": r.get("game_id"),
                        "game_name": r.get("game_name"),
                        "genre": r.get("genre"),
                        "stories": r.get("stories"),
                        "score": r.get("score"),
                    }
                )

        if not records:
            print("추천 결과가 없습니다. (records 비어 있음)")
            return

        df_out = pd.DataFrame(records)

        # 엑셀로 저장 (같은 폴더에 생성됨)
        df_out.to_excel(output_path, index=False)
        print(f"✅ 추천 결과 엑셀 파일 생성 완료: {output_path}")


# ---------------- 메인 실행부 ----------------

if __name__ == "__main__":
    # 같은 폴더에 있는 원본 데이터 파일 이름
    excel_path = "game_user_dataset.xlsx"

    # 추천기 생성
    recommender = GameRecommenderFromFile(excel_path)

    # 모든 유저에 대해 상위 5개 추천 → 엑셀로 저장
    output_file = "user_game_recommendations.xlsx"
    recommender.export_all_recommendations(
        output_path=output_file,
        top_k=5,          # 유저당 추천 게임 개수
        w_user=0.6,
        w_last=0.2,
        w_pop=0.2,
    )
