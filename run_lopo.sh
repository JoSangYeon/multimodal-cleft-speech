#!/usr/bin/env bash
# LOPO (leave-one-patient-out) 재실험 — Reviewer 2 코멘트 1 대응
#
# 34 folds x 7 modality combinations = 238 trainings.
# 결과는 전부 동일한 코드 버전에서 나와야 하므로 부분 재사용하지 않는다
# (완료된 조합은 건너뛰지만, 코드를 바꿨다면 출력 디렉토리를 새로 잡을 것).
#
# 적용된 수정:
#   - pos_weight를 fold별 학습셋에서 계산 (전역 고정값은 스케일 드리프트를 유발)
#   - 비활성 모달리티 미로딩
#   - VFS 임베딩 GPU 상주 캐시
#
# 사용법:
#   bash run_lopo.sh 0    # GPU 0 담당
#   bash run_lopo.sh 1    # GPU 1 담당

set -u
GPU=${1:?"GPU index (0 or 1) 를 지정하세요"}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SP2

ROOT=./RESULT_LOPO_V2
COMMON="--cv_scheme lopo --save_model False --pos_weight_mode fold \
        --dsr_gpu_cache True --save_root ${ROOT} \
        --epochs 8 --batch_size 32 --learning_rate 5e-6 --accum_step 1 --dsr_k 8"

# VFS 조합 ~1.5분/fold, 나머지 0.6~1.1분/fold 기준으로 두 GPU 부하를 맞춘다.
#
# name  audio video vfs tabular
case "${GPU}" in
  0) CONFIGS=(                                    # ~2.0시간
       "Audio_VFS           True False True  False"
       "Audio_Video_VFS     True True  True  False"
       "Audio_Only          True False False False"
     ) ;;
  1) CONFIGS=(                                    # ~2.4시간
       "Full_Modality       True True  True  True"
       "Audio_Video         True True  False False"
       "Audio_Video_Tabular True True  False True"
       "Audio_Tabular       True False False True"
     ) ;;
  *) echo "GPU는 0 또는 1이어야 합니다"; exit 1 ;;
esac

mkdir -p "${ROOT}/_logs"
echo "[GPU ${GPU}] ${#CONFIGS[@]}개 조합 시작 — $(date '+%F %T')"

for cfg in "${CONFIGS[@]}"; do
  read -r NAME A V D T <<< "${cfg}"
  LOG="${ROOT}/_logs/${NAME}.log"

  if [ -f "${ROOT}/${NAME}/inference_results_보상조음.csv" ]; then
    echo "[GPU ${GPU}] ${NAME} — 이미 완료됨, 건너뜀"
    continue
  fi

  echo "[GPU ${GPU}] ${NAME} 시작 — $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=${GPU} python main.py ${COMMON} \
    --save_dir "${NAME}" \
    --is_audio_active   "${A}" \
    --is_video_active   "${V}" \
    --is_dsr_active     "${D}" \
    --is_tabular_active "${T}" \
    > "${LOG}" 2>&1

  if [ $? -ne 0 ]; then
    echo "[GPU ${GPU}] ${NAME} 실패 — ${LOG} 확인"
  else
    echo "[GPU ${GPU}] ${NAME} 완료 — $(grep '총 소요' "${LOG}" || echo '')"
  fi
done

echo "[GPU ${GPU}] 전체 완료 — $(date '+%F %T')"
