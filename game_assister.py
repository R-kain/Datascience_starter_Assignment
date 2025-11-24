import numpy as np
from typing import List, Dict, Optional

# -------------------------------
# 1. 카테고리 정의
# -------------------------------

AGE_BUCKETS = ["10s", "20s", "30s", "40s", "50s+"]  # 10대, 20대, ...
GENDERS = ["M", "F"]

GENRES = [
    "RPG", "FPS", "AOS", "Hyper-FPS", "Roguelike",
    "Battle royale", "Survival", "Visual novel",
    "Horror", "Sport", "Rhythm", "Racing"
]

STORIES = [
    "fantasy", "mystery", "healing", "SF",
    "romance", "modern", "period", "action",
    "war", "sport"
]


# -------------------------------
# 2. 유틸 함수: 인덱스 매핑
# -------------------------------

age2idx = {a: i for i, a in enumerate(AGE_BUCKETS)}
gender2idx = {g: i for i, g in enumerate(GENDERS)}
genre2idx = {g: i for i, g in enumerate(GENRES)}
story2idx = {s: i for i, s in enumerate(STORIES)}


# -------------------------------
# 3. 벡터 인코딩 함수
# -------------------------------

def encode_age_bucket(age: int) -> np.ndarray:
    """
    실제 나이(int)를 받아서 AGE_BUCKETS 기준 one-hot으로 변환.
    """
    if age < 20:
        bucket = "10s"
    elif age < 30:
        bucket = "20s"
    elif age < 40:
        bucket = "30s"
    elif age < 50:
        bucket = "40s"
    else:
        bucket = "50s+"

    vec = np.zeros(len(AGE_BUCKETS), dtype=float)
    vec[age2idx[bucket]] = 1.0
    return vec


def encode_age_bucket_from_bucket(bucket: str) -> np.ndarray:
    """
    이미 "20s", "30s" 같은 버킷 문자열이 있을 때.
    """
    vec = np.zeros(len(AGE_BUCKETS), dtype=float)
    if bucket in age2idx:
        vec[age2idx[bucket]] = 1.0
    return vec


def encode_gender(gender: str) -> np.ndarray:
    """
    gender: "M" 또는 "F"
    """
    vec = np.zeros(len(GENDERS), dtype=float)
    if gender in gender2idx:
        vec[gender2idx[gender]] = 1.0
    return vec


def encode_gender_ratio(male_ratio: float) -> np.ndarray:
    """
    게임 이용자 성별 비율을 [male_ratio, female_ratio]로 표현.
    male_ratio: 0~1
    """
    male = np.clip(male_ratio, 0.0, 1.0)
    female = 1.0 - male
    return np.array([male, female], dtype=float)


def encode_multi_hot(items: List[str], mapping: Dict[str, int], size: int, normalize=True) -> np.ndarray:
    """
    장르/스토리처럼 여러 개 선택 가능한 것을 multi-hot으로 인코딩.
    normalize=True면 합이 1이 되도록 정규화.
    """
    vec = np.zeros(size, dtype=float)
    for item in items:
        if item in mapping:
            vec[mapping[item]] = 1.0
    if normalize and vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def encode_single_hot(item: str, mapping: Dict[str, int], size: int) -> np.ndarray:
    """
    장르처럼 하나만 선택하는 경우 (one-hot).
    """
    vec = np.zeros(size, dtype=float)
    if item in mapping:
        vec[mapping[item]] = 1.0
    return vec


# -------------------------------
# 4. 유저/게임 벡터 구조
#    - 유사도 계산에 사용할 공통 공간
#      [age_bucket, gender, genres, stories]
# -------------------------------

SIM_DIM = len(AGE_BUCKETS) + len(GENDERS) + len(GENRES) + len(STORIES)


def encode_user_for_similarity(user_profile: Dict) -> np.ndarray:
    """
    user_profile 예시:
    {
        "age": 22,
        "gender": "M",
        "preferred_genres": ["RPG", "Roguelike"],
        "preferred_stories": ["fantasy", "romance"]
    }
    """
    # 1) 나이 버킷
    age_vec = encode_age_bucket(user_profile["age"])

    # 2) 성별
    gender_vec = encode_gender(user_profile["gender"])

    # 3) 선호 장르 (multi-hot 정규화)
    pref_genres = user_profile.get("preferred_genres", [])
    genre_vec = encode_multi_hot(pref_genres, genre2idx, len(GENRES), normalize=True)

    # 4) 선호 스토리 (multi-hot 정규화)
    pref_stories = user_profile.get("preferred_stories", [])
    story_vec = encode_multi_hot(pref_stories, story2idx, len(STORIES), normalize=True)

    return np.concatenate([age_vec, gender_vec, genre_vec, story_vec])


def encode_game_for_similarity(game_info: Dict) -> np.ndarray:
    """
    game_info 예시:
    {
        "game_id": "game_1",
        "name": "Awesome RPG",
        "genre": "RPG",                  # 단일 장르
        "stories": ["fantasy", "war"],   # 다중 스토리
        "avg_user_age": 24.7,            # 평균 이용자 나이
        "male_ratio": 0.7,               # 남성 비율 (0~1)
        "avg_user_count": 12345          # (여기선 유사도에 안 쓰고 popularity에 사용)
    }
    """
    # 1) 평균 이용자 나이 → 버킷
    age_vec = encode_age_bucket(int(game_info["avg_user_age"]))

    # 2) 성별 비율
    gender_vec = encode_gender_ratio(game_info["male_ratio"])

    # 3) 장르 (one-hot)
    genre_vec = encode_single_hot(game_info["genre"], genre2idx, len(GENRES))

    # 4) 스토리 (multi-hot, 정규화 X / 그냥 존재만 표시)
    stories = game_info.get("stories", [])
    story_vec = encode_multi_hot(stories, story2idx, len(STORIES), normalize=False)

    return np.concatenate([age_vec, gender_vec, genre_vec, story_vec])


# -------------------------------
# 5. 코사인 유사도
# -------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -------------------------------
# 6. 추천 시스템 클래스
# -------------------------------

class GameRecommender:
    def __init__(self, games: List[Dict]):
        """
        games: 각 게임 정보 딕셔너리 리스트
        (encode_game_for_similarity 형식 참고)
        """
        self.games = games
        self.game_vectors = {}
        self.popularity = {}
        self._build(games)

    def _build(self, games: List[Dict]):
        # 1) 게임 벡터 미리 인코딩
        for g in games:
            gid = g["game_id"]
            self.game_vectors[gid] = encode_game_for_similarity(g)

        # 2) 인기 점수(popularity) 정규화
        counts = np.array([g.get("avg_user_count", 0) for g in games], dtype=float)
        if len(counts) > 0:
            min_c, max_c = counts.min(), counts.max()
            if max_c == min_c:
                norm_counts = np.ones_like(counts)
            else:
                norm_counts = (counts - min_c) / (max_c - min_c)
        else:
            norm_counts = np.array([])

        for g, c in zip(games, norm_counts):
            gid = g["game_id"]
            self.popularity[gid] = float(c)

    def recommend(
        self,
        user_profile: Dict,
        last_game_id: Optional[str] = None,
        top_k: int = 10,
        w_user: float = 0.6,
        w_last: float = 0.2,
        w_pop: float = 0.2,
    ) -> List[Dict]:
        """
        유저 프로필과 (선택) 마지막 플레이 게임을 기반으로
        상위 top_k개 게임을 추천.

        w_user + w_last + w_pop ≈ 1 이도록 사용하는 걸 권장.
        last_game_id가 없으면 w_last는 무시됨.
        """
        user_vec = encode_user_for_similarity(user_profile)

        # 마지막 게임 벡터 (있다면)
        last_vec = None
        if last_game_id is not None and last_game_id in self.game_vectors:
            last_vec = self.game_vectors[last_game_id]
        else:
            # last_game 없으면 last weight를 0으로
            w_user = w_user + w_last
            w_last = 0.0

        scores = []
        for g in self.games:
            gid = g["game_id"]
            gvec = self.game_vectors[gid]

            # 1) 유저-게임 유사도
            sim_user = cosine_similarity(user_vec, gvec)

            # 2) 마지막 게임과 유사도
            sim_last = 0.0
            if last_vec is not None:
                sim_last = cosine_similarity(last_vec, gvec)

            # 3) 인기 점수
            pop = self.popularity.get(gid, 0.0)

            # 4) 최종 점수
            score = w_user * sim_user + w_last * sim_last + w_pop * pop
            scores.append((score, g))

        # 점수 기준 내림차순 정렬
        scores.sort(key=lambda x: x[0], reverse=True)

        # 상위 top_k 게임 정보 + 점수 리턴
        result = []
        for s, g in scores[:top_k]:
            item = g.copy()
            item["score"] = s
            result.append(item)
        return result
