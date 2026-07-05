"""
Week 2 - Day 2 : Canny 엣지 파라미터 튜닝
AI허브 주행 프레임 데이터를 대상으로 Canny/blur 파라미터를 실험하고
최적값을 여러 프레임에서 검증한 뒤 결과를 저장한다.

실행 방법:
    python week2_day2_canny_tuning.py

각 단계는 함수로 분리되어 있고, 맨 아래 main에서 순서대로 호출한다.
필요 없는 단계는 main에서 주석 처리하면 된다.
"""

import cv2
import glob
import os

# ─────────────────────────────────────────────
# 설정 (여기만 네 환경에 맞게 수정)
# ─────────────────────────────────────────────
FRAME_DIR = "도로주행영상/Validation/원천데이터/bb/자동차전용도로/주간일출/맑음/30_전방/20200910_104924/1/"       # 프레임 이미지가 든 폴더
FRAME_EXT = "*.jpg"              # 확장자 (.png 이면 "*.png")
BLUR_KSIZE = 9                   # Gaussian Blur 커널 크기 (홀수)
OUTPUT_DIR = "results"           # 결과 저장 폴더

# 트랙바에서 찾은 최적값을 여기에 기록 (단계 2 실행 후 갱신)
BEST_LOW = 50
BEST_HIGH = 100


def load_frame_paths():
    """폴더에서 프레임 이미지 경로를 정렬해 가져온다."""
    paths = sorted(glob.glob(os.path.join(FRAME_DIR, FRAME_EXT)))
    print(f"[load] 찾은 이미지 수: {len(paths)}")
    if len(paths) == 0:
        print("[load] 경고: 이미지를 못 찾음. FRAME_DIR / FRAME_EXT 를 확인하세요.")
    return paths


# ─────────────────────────────────────────────
# 단계 1 : 1주차 파이프라인을 실제 프레임에 적용해 본다
#          (기본값이 실제 데이터에서 어떻게 무너지는지 관찰)
# ─────────────────────────────────────────────
def step1_baseline(paths):
    print("\n[단계1] 1주차 기본값(100,200)으로 첫 프레임 적용")
    img = cv2.imread(paths[0])
    print("        이미지 크기:", img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(blur, 100, 200)

    cv2.imshow("step1 - original", img)
    cv2.imshow("step1 - edges (100,200)", edges)
    print("        아무 키나 누르면 다음 단계로 넘어갑니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 2 : 트랙바로 Canny 임계값 실시간 튜닝 (오늘의 핵심)
#          슬라이더를 움직여 최적 low/high 를 찾는다.
# ─────────────────────────────────────────────
def step2_trackbar(paths):
    print("\n[단계2] 트랙바 튜닝 - 슬라이더로 low/high 조절, q 로 종료")
    img = cv2.imread(paths[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    def nothing(x):
        pass

    cv2.namedWindow("canny")
    cv2.createTrackbar("low",  "canny", 50,  255, nothing)
    cv2.createTrackbar("high", "canny", 150, 255, nothing)

    while True:
        low  = cv2.getTrackbarPos("low",  "canny")
        high = cv2.getTrackbarPos("high", "canny")
        edges = cv2.Canny(blur, low, high)

        # 현재 값을 화면에 표시
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, f"low={low} high={high}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("canny", display)

        if cv2.waitKey(1) == ord('q'):
            print(f"        최종 선택값 → low={low}, high={high}")
            print("        이 값을 파일 상단 BEST_LOW / BEST_HIGH 에 기록하세요.")
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 3 : blur 강도별 비교 (노이즈 제거 vs 차선 보존)
# ─────────────────────────────────────────────
def step3_compare_blur(paths):
    print("\n[단계3] blur 커널 크기별 비교 (3, 5, 9)")
    img = cv2.imread(paths[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for k in [3, 5, 9]:
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blur, BEST_LOW, BEST_HIGH)
        cv2.imshow(f"step3 - blur {k}x{k}", edges)

    print("        아무 키나 누르면 다음 단계로.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 4 : 최적값을 여러 프레임에서 검증
#          한 프레임 최적 != 전체 최적. 일반적으로 잘 되는지 확인.
# ─────────────────────────────────────────────
def step4_validate(paths, n=10, delay_ms=500):
    print(f"\n[단계4] 앞 {n}개 프레임에 최적값 적용 (low={BEST_LOW}, high={BEST_HIGH})")
    print("        q 로 중단")
    for path in paths[:n]:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
        edges = cv2.Canny(blur, BEST_LOW, BEST_HIGH)
        cv2.imshow("step4 - validate", edges)
        if cv2.waitKey(delay_ms) == ord('q'):
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 단계 5 : 대표 프레임 결과 저장 (보고서/블로그용)
# ─────────────────────────────────────────────
def step5_save(paths, n=3):
    print(f"\n[단계5] 대표 프레임 {n}장 결과 저장 → '{OUTPUT_DIR}/'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, path in enumerate(paths[:n]):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
        edges = cv2.Canny(blur, BEST_LOW, BEST_HIGH)

        # 원본과 엣지를 나란히 붙여 저장 (비교용)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([img, edges_bgr])
        out_path = os.path.join(OUTPUT_DIR, f"result_{i}.jpg")
        cv2.imwrite(out_path, combined)
        print(f"        저장: {out_path}")


# ─────────────────────────────────────────────
# main : 단계별 순차 실행
#        필요 없는 단계는 주석 처리하면 됨
# ─────────────────────────────────────────────
if __name__ == "__main__":
    paths = load_frame_paths()

    if len(paths) > 0:
        #step1_baseline(paths)        # 기본값 관찰
        #step2_trackbar(paths)        # 트랙바로 최적값 탐색  ← 여기서 찾은 값을 상단에 기록
        #step3_compare_blur(paths)    # blur 비교
        step4_validate(paths)        # 여러 프레임 검증
        #step5_save(paths)            # 결과 저장

    print("\n[done] 종료")