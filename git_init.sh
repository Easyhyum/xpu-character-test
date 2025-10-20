#!/bin/bash

echo "=== Git 초기화 및 설정 스크립트 ==="

# Git global 사용자 정보 설정
echo "Git 사용자 정보 설정 중..."
git config --global user.name "easyhyum"
git config --global user.email "easyhyum@naver.com"

echo "✓ Git 사용자 정보 설정 완료:"
echo "  - Name: $(git config --global user.name)"
echo "  - Email: $(git config --global user.email)"

# 현재 디렉토리에 Git 저장소 초기화
echo "Git 저장소 초기화 중..."
git init

# 원격 저장소 연결
echo "원격 저장소 연결 중..."
git remote add origin https://github.com/Easyhyum/xpu-character-test.git

# 원격 저장소에서 최신 내용 가져오기
echo "원격 저장소 내용 가져오는 중..."
git fetch origin

# main 브랜치로 체크아웃 및 추적 설정
echo "main 브랜치 설정 중..."
git checkout -b main
git branch --set-upstream-to=origin/main main

echo "✓ Git 초기화 및 원격 저장소 연결 완료!"
echo "✓ Repository: https://github.com/Easyhyum/xpu-character-test.git"
echo ""
echo "다음 명령어로 변경사항을 확인할 수 있습니다:"
echo "  git status"
echo "  git log --oneline"