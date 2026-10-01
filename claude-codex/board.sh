#!/usr/bin/env bash
# 四个窗口之间的锁与留言板（用法见 README.md §3、§4）
#
#   board.sh run   <锁名> <线> -- <命令...>    持锁执行命令，结束自动释放
#   board.sh lock  <锁名> <线>                 取锁（最多等 LOCK_WAIT_S 秒，默认 600），之后用 Edit 改文件
#   board.sh unlock <锁名> <线>                释放自己持有的锁
#   board.sh locks                             列出当前被持有的锁
#   board.sh post  <消息|待决> <发件> <收件> <类型> <标题>   正文从 stdin 读，自动编号并打印编号
#   board.sh inbox <线> [起始编号]             列出发给该线或全体、编号大于起始编号的消息
#
# 线名：L1 L2 L3 L4 user；收件还可以是「全体」。锁名：小写字母、数字、-、_。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCKS="$DIR/.locks"
WAIT_S="${LOCK_WAIT_S:-600}"
mkdir -p "$LOCKS"

die() { echo "board.sh: $*" >&2; exit 2; }
check_name() { [[ "$1" =~ ^[a-z][a-z0-9_-]*$ ]] || die "锁名只能用小写字母、数字、-、_：$1"; }
check_line() { [[ "$1" =~ ^(L[1-4]|user)$ ]] || die "线名必须是 L1~L4 或 user：$1"; }

# owner 文件一行：<线> <令牌> <时间>。run / post 的令牌是本次调用独有的，lock 的令牌是 manual。
owner_of() { [[ -f "$LOCKS/$1/owner" ]] && cat "$LOCKS/$1/owner" || echo "?"; }

# 「查锁目录 + 建目录 + 写 owner」和「核对 owner + 删目录」都在 .guard 的 flock 里做。
# 不能只靠 mkdir 的原子性：本机的 mkdir 是 uutils coreutils，并发时会有多个进程同时成功。
guarded() { flock "$LOCKS/.guard" "$@"; }

# acquire <锁名> <线> <令牌>：拿不到就一直等，同一条线的两次调用也互斥
acquire() {
  local name=$1 line=$2 token=$3 d="$LOCKS/$1" start
  start=$(date +%s)
  until guarded bash -c '[[ ! -e "$1" ]] && mkdir "$1" && printf "%s\n" "$2" > "$1/owner"' \
        _ "$d" "$line $token $(date '+%Y-%m-%d %H:%M:%S')"; do
    if (( $(date +%s) - start >= WAIT_S )); then
      echo "board.sh: 等锁 $name 超过 ${WAIT_S}s，持有者：$(owner_of "$name")" >&2
      echo "board.sh: 给持有者发消息确认；持有者窗口已经不在时请用户删除 $d" >&2
      exit 1
    fi
    sleep 0.2
  done
}

# release <锁名> <线> <令牌>：只有持有者本人（线和令牌都对上）能释放
release() {
  local name=$1 line=$2 token=$3 d="$LOCKS/$1" rc=0
  guarded bash -c '
    [[ -d "$1" ]] || { echo "board.sh: 锁 $2 没有被持有" >&2; exit 0; }
    who=$(cat "$1/owner" 2>/dev/null || echo "?")
    if [[ "$(cut -d" " -f1-2 <<<"$who")" != "$3 $4" ]]; then
      echo "board.sh: 锁 $2 由 $who 持有，$3 不能释放" >&2
      exit 2
    fi
    rm -f "$1/owner" && rmdir "$1"
  ' _ "$d" "$name" "$line" "$token" || rc=$?
  return "$rc"
}

new_token() { echo "$$-$RANDOM$RANDOM"; }

board_file() {
  case "$1" in
    消息) echo "$DIR/消息.md" ;;
    待决) echo "$DIR/待决.md" ;;
    *) die "只能 post 到 消息 或 待决：$1" ;;
  esac
}

cmd=${1:-}
[[ -n "$cmd" ]] || die "缺少子命令（run / lock / unlock / locks / post / inbox）"
shift

case "$cmd" in
  run)
    [[ $# -ge 4 && "$3" == "--" ]] || die "用法：board.sh run <锁名> <线> -- <命令...>"
    name=$1 line=$2; shift 3
    check_name "$name"; check_line "$line"
    if [[ "$(owner_of "$name" | cut -d' ' -f1-2)" == "$line manual" ]]; then
      "$@"   # 本线已经用 lock 手动持有这把锁：直接执行，不重复取锁也不释放
    else
      token=$(new_token)
      acquire "$name" "$line" "$token"
      trap 'release "$name" "$line" "$token"' EXIT
      "$@"
    fi
    ;;
  lock)
    [[ $# -eq 2 ]] || die "用法：board.sh lock <锁名> <线>"
    check_name "$1"; check_line "$2"
    if [[ "$(owner_of "$1" | cut -d' ' -f1-2)" == "$2 manual" ]]; then
      echo "锁 $1 本来就由 $2 持有（$(owner_of "$1")）"
    else
      acquire "$1" "$2" manual
      echo "已持有锁 $1；改完立刻 board.sh unlock $1 $2"
    fi
    ;;
  unlock)
    [[ $# -eq 2 ]] || die "用法：board.sh unlock <锁名> <线>"
    check_name "$1"; check_line "$2"
    release "$1" "$2" manual || exit $?
    ;;
  locks)
    found=0
    for d in "$LOCKS"/*/; do
      [[ -d "$d" ]] || continue
      found=1
      echo "$(basename "$d")  $(owner_of "$(basename "$d")")"
    done
    [[ $found -eq 1 ]] || echo "（没有被持有的锁）"
    ;;
  post)
    [[ $# -eq 5 ]] || die "用法：board.sh post <消息|待决> <发件> <收件> <类型> <标题>（正文走 stdin）"
    board=$1 from=$2 to=$3 kind=$4 title=$5
    check_line "$from"
    [[ "$to" =~ ^(L[1-4]|user|全体)$ ]] || die "收件必须是 L1~L4、user 或 全体：$to"
    [[ "$title" != *$'\n'* ]] || die "标题不能换行"
    file=$(board_file "$board")
    body=$(cat)
    [[ -n "${body//[[:space:]]/}" ]] || die "正文为空（用 heredoc 从 stdin 传入）"
    lockname="board-$([[ $board == 消息 ]] && echo msg || echo decision)"
    token=$(new_token)
    acquire "$lockname" "$from" "$token"
    trap 'release "$lockname" "$from" "$token"' EXIT
    if [[ $board == 消息 ]]; then
      last=$(grep -oE '^### #[0-9]+' "$file" | tail -1 | grep -oE '[0-9]+' || true)
      id=$(printf '#%03d' $((10#${last:-0} + 1)))
    else
      last=$(grep -oE '^### D-[0-9]+' "$file" | tail -1 | grep -oE '[0-9]+' || true)
      id=$(printf 'D-%03d' $((10#${last:-0} + 1)))
    fi
    {
      printf '\n### %s · %s → %s · %s · %s\n' "$id" "$from" "$to" "$kind" "$title"
      printf '_%s_\n\n' "$(date '+%Y-%m-%d %H:%M')"
      printf '%s\n' "$body"
    } >> "$file"
    echo "$id"
    ;;
  inbox)
    [[ $# -ge 1 && $# -le 2 ]] || die "用法：board.sh inbox <线> [起始编号，如 #012 或 12]"
    check_line "$1"
    after=$(echo "${2:-0}" | grep -oE '[0-9]+' || echo 0)
    awk -v me="$1" -v after="$((10#$after))" '
      /^### #[0-9]+ / {
        n = substr($2, 2) + 0
        split($0, f, " · ")
        split(f[2], ft, " → ")
        show = (n > after) && (ft[2] == me || ft[2] == "全体") && ft[1] != me
      }
      show { print }
    ' "$DIR/消息.md"
    ;;
  *)
    die "未知子命令：$cmd"
    ;;
esac
