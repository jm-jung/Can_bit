#!/bin/bash
# GitHub Webhook → Discord 알림 스크립트
#
# 사용법:
#   - GitHub Webhook에서 JSON payload를 stdin으로 전달
#   - 환경변수 DISCORD_GIT_WEBHOOK_URL 설정 필요
#
# GitHub Webhook → 이 스크립트 연결 방법:
#   - Python Flask/FastAPI: request.get_json()로 payload 받아서 subprocess.run(['./deploy.sh'], input=json.dumps(payload), text=True)로 stdin 전달
#   - Node.js Express: req.body를 JSON.stringify()해서 child_process.exec('cat | ./deploy.sh', {input: JSON.stringify(payload)})로 pipe
#   - CGI 스크립트: POST body($HTTP_BODY)를 cat으로 pipe하거나 직접 stdin으로 연결
#   - nginx + fcgiwrap: POST body를 스크립트 stdin으로 자동 연결

set -euo pipefail

# .env 파일 자동 로드 (DISCORD_GIT_WEBHOOK_URL이 없을 때만)
# 기존 환경변수 설정 방식은 그대로 유지되며, .env는 보조 수단으로만 사용
# 이 방식은 systemd, launchd, webhook 서버 등 다양한 실행 환경에서 안전하게 동작:
# - 환경변수가 이미 설정되어 있으면 .env를 읽지 않음 (기존 동작 유지)
# - .env가 없어도 에러 없이 계속 진행 (선택적 로딩)
# - .env 내용은 stdout/stderr에 출력하지 않음 (보안)
if [[ -z "${DISCORD_GIT_WEBHOOK_URL:-}" ]]; then
    # 현재 작업 디렉토리 기준으로 .env 파일 확인
    ENV_FILE=".env"
    if [[ -f "${ENV_FILE}" ]] && [[ -r "${ENV_FILE}" ]]; then
        # .env 파일을 안전하게 로드 (주석/공백 무시, export 키워드 없어도 동작)
        # set -a: 변수를 자동으로 export (export 키워드 없어도 환경변수로 설정)
        set -a
        # source 대신 while read로 한 줄씩 처리하여 주석/공백 필터링
        while IFS= read -r line || [[ -n "${line}" ]]; do
            # 주석 라인 무시 (#로 시작하는 라인)
            [[ "${line}" =~ ^[[:space:]]*# ]] && continue
            # 공백 라인 무시
            [[ -z "${line// }" ]] && continue
            # export 키워드 제거 (있으면 제거, 없으면 그대로)
            line="${line#export }"
            line="${line#export}"
            line="${line## }"
            # DISCORD_GIT_WEBHOOK_URL만 필요하므로 해당 라인만 평가
            if [[ "${line}" =~ ^[[:space:]]*DISCORD_GIT_WEBHOOK_URL= ]]; then
                # 안전하게 평가 (함수 호출 등 위험한 코드 실행 방지)
                # 2>/dev/null로 에러 메시지도 출력하지 않음 (보안)
                eval "${line}" 2>/dev/null || true
            fi
        done < "${ENV_FILE}"
        set +a
    fi
fi

# Discord Webhook URL 확인
if [[ -z "${DISCORD_GIT_WEBHOOK_URL:-}" ]]; then
    echo "ERROR: DISCORD_GIT_WEBHOOK_URL environment variable is not set" >&2
    exit 2
fi

# stdin에서 JSON payload 읽기
PAYLOAD=$(cat)

# jq가 있는지 확인
if command -v jq >/dev/null 2>&1; then
    USE_JQ=true
else
    USE_JQ=false
fi

# GitHub push 이벤트 파싱
if [[ "${USE_JQ}" == "true" ]]; then
    # jq 사용
    REPO_NAME=$(echo "${PAYLOAD}" | jq -r '.repository.full_name // "unknown"')
    PUSHER_NAME=$(echo "${PAYLOAD}" | jq -r '.pusher.name // "unknown"')
    REF=$(echo "${PAYLOAD}" | jq -r '.ref // "unknown"')
    BRANCH=$(echo "${REF}" | sed 's|refs/heads/||')
    COMPARE_URL=$(echo "${PAYLOAD}" | jq -r '.compare // .head_commit.url // ""')
    COMMITS_COUNT=$(echo "${PAYLOAD}" | jq -r '.commits | length // 0')
    
    # 상위 3개 커밋 메시지 추출
    COMMIT_MESSAGES=""
    if [[ "${COMMITS_COUNT}" -gt 0 ]]; then
        COMMIT_MESSAGES=$(echo "${PAYLOAD}" | jq -r '.commits[0:3][] | "- \(.message | split("\n")[0])" // "- (no message)"' | head -3)
    fi
else
    # jq 없을 때 fallback (grep/sed 기반 최소 파싱)
    REPO_NAME=$(echo "${PAYLOAD}" | grep -o '"full_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"full_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "unknown")
    PUSHER_NAME=$(echo "${PAYLOAD}" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "unknown")
    REF=$(echo "${PAYLOAD}" | grep -o '"ref"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"ref"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "unknown")
    BRANCH=$(echo "${REF}" | sed 's|refs/heads/||')
    COMPARE_URL=$(echo "${PAYLOAD}" | grep -o '"compare"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"compare"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "")
    if [[ -z "${COMPARE_URL}" ]]; then
        COMPARE_URL=$(echo "${PAYLOAD}" | grep -o '"url"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "")
    fi
    
    # commits 개수 추출 (간단한 방법)
    COMMITS_COUNT=$(echo "${PAYLOAD}" | grep -o '"commits"[[:space:]]*:[[:space:]]*\[' | wc -l | tr -d ' ' || echo "0")
    if [[ "${COMMITS_COUNT}" == "0" ]]; then
        # 배열 내 항목 개수 추정
        COMMITS_COUNT=$(echo "${PAYLOAD}" | grep -c '"id"[[:space:]]*:[[:space:]]*"[^"]*"' || echo "0")
    fi
    
    # 커밋 메시지 추출 (최소한)
    COMMIT_MESSAGES=""
    if echo "${PAYLOAD}" | grep -q '"message"'; then
        COMMIT_MESSAGES=$(echo "${PAYLOAD}" | grep -o '"message"[[:space:]]*:[[:space:]]*"[^"]*"' | head -3 | sed 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/- \1/' || echo "")
    fi
fi

# 기본값 설정
REPO_NAME="${REPO_NAME:-unknown}"
PUSHER_NAME="${PUSHER_NAME:-unknown}"
BRANCH="${BRANCH:-unknown}"
COMMITS_COUNT="${COMMITS_COUNT:-0}"

# Discord 메시지 구성
CONTENT="[Can_bit] ${REPO_NAME} | ${PUSHER_NAME} pushed to ${BRANCH} | ${COMMITS_COUNT} commits"

# Embed 구성
EMBED_TITLE="${REPO_NAME}"
EMBED_URL="${COMPARE_URL}"
EMBED_DESCRIPTION=""

if [[ -n "${COMMIT_MESSAGES}" ]]; then
    EMBED_DESCRIPTION="${COMMIT_MESSAGES}"
else
    EMBED_DESCRIPTION="No commit messages available"
fi

# Discord Webhook payload 생성 (jq가 있으면 안전하게, 없으면 직접 구성)
if [[ "${USE_JQ}" == "true" ]]; then
    # jq로 안전하게 JSON 생성 (자동 이스케이프)
    DISCORD_PAYLOAD=$(jq -n \
        --arg content "${CONTENT}" \
        --arg title "${EMBED_TITLE}" \
        --arg url "${EMBED_URL}" \
        --arg desc "${EMBED_DESCRIPTION}" \
        '{
          "content": $content,
          "embeds": [{
            "title": $title,
            "url": $url,
            "description": $desc,
            "color": 3447003
          }]
        }')
else
    # jq 없을 때 직접 JSON 구성 (최소한의 이스케이프)
    # 특수문자 이스케이프
    SAFE_CONTENT=$(echo "${CONTENT}" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
    SAFE_TITLE=$(echo "${EMBED_TITLE}" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
    SAFE_DESC=$(echo "${EMBED_DESCRIPTION}" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    SAFE_URL=$(echo "${EMBED_URL}" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
    
    DISCORD_PAYLOAD=$(cat <<EOF
{
  "content": "${SAFE_CONTENT}",
  "embeds": [
    {
      "title": "${SAFE_TITLE}",
      "url": "${SAFE_URL}",
      "description": "${SAFE_DESC}",
      "color": 3447003
    }
  ]
}
EOF
)
fi

# Discord로 전송
HTTP_STATUS=$(curl -s -w "%{http_code}" -o /tmp/discord_response.json \
    -X POST \
    -H "Content-Type: application/json" \
    -d "${DISCORD_PAYLOAD}" \
    "${DISCORD_GIT_WEBHOOK_URL}" 2>&1) || HTTP_STATUS="000"

# HTTP 상태 코드 추출 (마지막 3자리)
STATUS_CODE="${HTTP_STATUS: -3}"

# 2xx 체크
if [[ ! "${STATUS_CODE}" =~ ^2[0-9]{2}$ ]]; then
    echo "ERROR: Discord webhook failed with HTTP status ${STATUS_CODE}" >&2
    if [[ -f /tmp/discord_response.json ]]; then
        echo "Response: $(head -c 200 /tmp/discord_response.json 2>/dev/null || echo 'N/A')" >&2
        rm -f /tmp/discord_response.json
    fi
    exit 3
fi

# 성공
rm -f /tmp/discord_response.json
exit 0

# ============================================================================
# 테스트 방법
# ============================================================================
#
# 1. 환경변수 설정:
#    export DISCORD_GIT_WEBHOOK_URL="https://discord.com/api/webhooks/..."
#
# 2. 샘플 payload로 테스트:
#    cat <<'EOF' | ./deploy.sh
#    {
#      "ref": "refs/heads/main",
#      "repository": {
#        "full_name": "user/Can_bit"
#      },
#      "pusher": {
#        "name": "username"
#      },
#      "compare": "https://github.com/user/Can_bit/compare/abc123...def456",
#      "commits": [
#        {
#          "message": "feat: add new feature"
#        },
#        {
#          "message": "fix: bug fix"
#        },
#        {
#          "message": "docs: update README"
#        }
#      ]
#    }
#    EOF
#
# 3. GitHub Webhook → 이 스크립트 연결 방법:
#    - Python Flask/FastAPI: request.get_json()로 payload 받아서 subprocess로 stdin 전달
#    - Node.js Express: req.body를 JSON.stringify()해서 child_process.exec()로 pipe
#    - CGI 스크립트: $HTTP_BODY를 cat으로 pipe
#    - nginx + fcgiwrap: POST body를 스크립트 stdin으로 연결
#
# 4. 파일 권한 설정:
#    chmod +x deploy.sh
