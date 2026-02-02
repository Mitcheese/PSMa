#!/usr/bin/env bash
# Mirror wwPDB 传统 PDB 文本（.ent.gz）→ 校验/修复 → 解压为 .pdb（成功即删源）→ 每 20,000 个分桶
# 适用：CentOS/通用Linux集群
# 依赖：rsync、gzip（可选：pigz 加速）
# 用法：chmod +x mirror_pdb_text.sh
#       ./mirror_pdb_text.sh /path/to/large_disk/pdb_text_archive
# 可选环境变量：BATCH_SIZE=20000 NPROC=<并行数> SKIP_RSYNC=0 CLEAN_FLAT_AFTER_CHUNK=0

set -euo pipefail

# ========= 配置 =========
ROOT_DIR="${1:-$PWD/pdb_text_archive}"
BATCH_SIZE="${BATCH_SIZE:-20000}"
NPROC="${NPROC:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"
SKIP_RSYNC="${SKIP_RSYNC:-0}"                    # 设为1可跳过Step1，仅做修复/解压/分桶
CLEAN_FLAT_AFTER_CHUNK="${CLEAN_FLAT_AFTER_CHUNK:-0}"  # 设为1在分桶后清空_flat_pdb

# 目录结构
WORK_DIVIDED="$ROOT_DIR/_divided_pdb_gz"   # rsync 下来的 .ent.gz（按官方divided目录）
FLAT_DIR="$ROOT_DIR/_flat_pdb"             # 解压后的 .pdb（临时平铺）
CHUNK_DIR="$ROOT_DIR/pdb_chunks"           # 分桶结果（每20000个）

mkdir -p "$WORK_DIVIDED" "$FLAT_DIR" "$CHUNK_DIR"

# 镜像源（优先使用 PDBe；RCSB 需自定义端口；PDBj 作为兜底）
MIRROR_PRIMARY="rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/pdb/"
MIRROR_ALT1="rsync.rcsb.org::ftp_data/structures/divided/pdb/"
MIRROR_ALT1_PORT=33444
MIRROR_ALT2="ftp.pdbj.org::ftp_data/structures/divided/pdb/"

# 通用 rsync 选项（append-verify + checksum 保证内容完整；partial 便于断点续传）
RSYNC_COMMON="-rlpt -z --append-verify --checksum --partial --partial-dir=.rsync-partial --timeout=60"

# 日志清单
BAD_LIST="$ROOT_DIR/.bad_gz.list"
REPAIRED_LIST="$ROOT_DIR/.repaired.list"
FAILED_REPAIR_LIST="$ROOT_DIR/.failed_repair.list"
FAILED_EXTRACT_LIST="$ROOT_DIR/.failed_extract.list"

# ========= 打印环境信息 =========
echo "==== wwPDB PDB 文本镜像脚本 ===="
echo "ROOT_DIR: $ROOT_DIR"
echo "NPROC: $NPROC"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SKIP_RSYNC: $SKIP_RSYNC"
echo "CLEAN_FLAT_AFTER_CHUNK: $CLEAN_FLAT_AFTER_CHUNK"
echo "---------------------------------"
df -h "$ROOT_DIR" 2>/dev/null || true
df -i "$ROOT_DIR" 2>/dev/null || true
( quota -s || true ) 2>/dev/null || true
echo "---------------------------------"

# ========= Step 1: 初次/增量同步（可跳过） =========
if [ "$SKIP_RSYNC" -eq 0 ]; then
  echo "== 1/3 同步 .ent.gz 到 $WORK_DIVIDED（来源：$MIRROR_PRIMARY） =="
  rsync $RSYNC_COMMON --delete "$MIRROR_PRIMARY" "$WORK_DIVIDED"
  echo "Step 1 完成。"
else
  echo "== 1/3 跳过同步（SKIP_RSYNC=1） =="
fi

# ========= 辅助函数 =========
# 多镜像精准重拉单个相对路径文件（相对 WORK_DIVIDED），成功返回0，失败返回1；并记录清单
rsync_fetch_rel() {
  local rel="$1"                          # e.g., lk/pdb9lk5.ent.gz
  local rel_dir; rel_dir="$(dirname "$rel")"
  mkdir -p "$WORK_DIVIDED/$rel_dir"

  # Primary
  if rsync $RSYNC_COMMON "${MIRROR_PRIMARY}./${rel}" "$WORK_DIVIDED/$rel_dir/" >/dev/null 2>&1; then
    echo "$rel" >> "$REPAIRED_LIST"; return 0
  fi
  # Alt1 (RCSB with custom port)
  if rsync $RSYNC_COMMON --port="$MIRROR_ALT1_PORT" "${MIRROR_ALT1}./${rel}" "$WORK_DIVIDED/$rel_dir/" >/dev/null 2>&1; then
    echo "$rel" >> "$REPAIRED_LIST"; return 0
  fi
  # Alt2 (PDBj)
  if rsync $RSYNC_COMMON "${MIRROR_ALT2}./${rel}" "$WORK_DIVIDED/$rel_dir/" >/dev/null 2>&1; then
    echo "$rel" >> "$REPAIRED_LIST"; return 0
  fi

  echo "$rel" >> "$FAILED_REPAIR_LIST"
  return 1
}

# 解压一个 .ent.gz → .pdb；失败则自动重拉再试一次；成功后删除源 .ent.gz
extract_one() {
  local src="$1"                       # 绝对路径
  local base rel id out
  base="$(basename "$src")"            # pdbXXXX.ent.gz
  rel="${src#"$WORK_DIVIDED"/}"        # 相对路径（lk/pdb9lk5.ent.gz）
  id="${base#pdb}"; id="${id%.ent.gz}" # XXXX
  out="$FLAT_DIR/${id}.pdb"

  do_extract() {
    if command -v pigz >/dev/null 2>&1; then
      pigz -dc "$src" > "$out"
    else
      gunzip -c "$src" > "$out"
    fi
  }

  # 第一次尝试解压
  if do_extract 2>/dev/null; then
    rm -f "$src"            # 成功即删源，降低占用
    return 0
  fi

  echo "解压失败，尝试重拉后再解压：$rel"
  if rsync_fetch_rel "$rel"; then
    if do_extract 2>/dev/null; then
      rm -f "$src"
      return 0
    fi
  fi

  echo "$rel" >> "$FAILED_EXTRACT_LIST"
  # 清理可能的半成品
  [ -f "$out" ] && rm -f "$out" || true
  return 1
}

export WORK_DIVIDED FLAT_DIR ROOT_DIR NPROC
export BAD_LIST REPAIRED_LIST FAILED_REPAIR_LIST FAILED_EXTRACT_LIST
export -f rsync_fetch_rel extract_one

# ========= Step 2: 校验 → 针对性修复 → 解压（成功即删源） =========
echo "== 2/3 校验 + 针对性修复 + 解压（失败自动再拉；成功即删源） =="

: > "$BAD_LIST"; : > "$REPAIRED_LIST"; : > "$FAILED_REPAIR_LIST"; : > "$FAILED_EXTRACT_LIST"

echo "扫描并校验 .ent.gz ..."
# 并行校验，收集坏件清单（相对路径）
find "$WORK_DIVIDED" -type f -name 'pdb????.ent.gz' -print0 \
| xargs -0 -n1 -P "$NPROC" bash -c '
  f="$1"
  if ! gzip -t "$f" 2>/dev/null; then
    echo "${f#'"$WORK_DIVIDED"'/}" >> "'"$BAD_LIST"'"
  fi
' _

BAD_CNT=$( [ -s "$BAD_LIST" ] && wc -l < "$BAD_LIST" || echo 0 )
echo "发现坏件/不完整：${BAD_CNT} 个"

# 针对性重拉坏件（多镜像兜底），失败不使主脚本中断
if [ "$BAD_CNT" -gt 0 ]; then
  echo "开始针对性重拉（含多镜像兜底）..."
  set +e
  sort -u "$BAD_LIST" \
  | xargs -r -n1 -P "$NPROC" -I{} bash -c 'rsync_fetch_rel "$@" || true' _ {} \
  || true
  set -e
  REPAIRED_CNT=$( [ -s "$REPAIRED_LIST" ] && wc -l < "$REPAIRED_LIST" || echo 0 )
  FAILED_REPAIR_CNT=$( [ -s "$FAILED_REPAIR_LIST" ] && wc -l < "$FAILED_REPAIR_LIST" || echo 0 )
  echo "重拉完成：修复 ${REPAIRED_CNT} 个，未能修复 ${FAILED_REPAIR_CNT} 个"
fi

echo "开始解压为 .pdb（失败自动再拉一次；成功即删源）..."
set +e
find "$WORK_DIVIDED" -type f -name 'pdb????.ent.gz' -print0 \
| xargs -0 -n1 -P "$NPROC" bash -c 'extract_one "$@" || true' _ \
|| true
set -e

FAILED_EXTRACT_CNT=$( [ -s "$FAILED_EXTRACT_LIST" ] && wc -l < "$FAILED_EXTRACT_LIST" || echo 0 )
echo "== Step 2 完成 =="
echo "初次坏件数: ${BAD_CNT}"
echo "已修复数: $( [ -s "$REPAIRED_LIST" ] && wc -l < "$REPAIRED_LIST" || echo 0 )"
echo "仍未修复: $( [ -s "$FAILED_REPAIR_LIST" ] && wc -l < "$FAILED_REPAIR_LIST" || echo 0 )"
echo "解压失败并仍未修复: ${FAILED_EXTRACT_CNT}"
[ "$FAILED_EXTRACT_CNT" -gt 0 ] && echo "清单见：$FAILED_EXTRACT_LIST"

# ========= Step 3: 分桶（每 20,000 个 .pdb） =========
echo "== 3/3 分桶：每桶 ${BATCH_SIZE} 个 .pdb 到 $CHUNK_DIR =="
shopt -s nullglob
count=0
batch=1
dest="$CHUNK_DIR/$(printf "set_%05d" "$batch")"
mkdir -p "$dest"

for f in "$FLAT_DIR"/*.pdb; do
  mv "$f" "$dest/"
  count=$((count+1))
  if (( count % BATCH_SIZE == 0 )); then
    batch=$((batch+1))
    dest="$CHUNK_DIR/$(printf "set_%05d" "$batch")"
    mkdir -p "$dest"
  fi
done
shopt -u nullglob

echo "分桶完成：共移动 ${count} 个文件。最终目录：$CHUNK_DIR"

if [ "$CLEAN_FLAT_AFTER_CHUNK" -eq 1 ]; then
  echo "清理 _flat_pdb（按 CLEAN_FLAT_AFTER_CHUNK=1 要求）..."
  rm -rf "$FLAT_DIR"
fi

echo "全部完成。"
echo "失败清单（如有）："
[ -s "$FAILED_REPAIR_LIST" ]  && echo " - 修复失败列表：$FAILED_REPAIR_LIST"
[ -s "$FAILED_EXTRACT_LIST" ] && echo " - 解压失败列表：$FAILED_EXTRACT_LIST"