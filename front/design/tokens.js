export const colors = {
  background: '#181A20',
  text: '#FFFFFF',
  textMuted: '#9EA3B2',
  circle: '#D9D9D9',
  tabBar: '#21232B',
  accentGreen: '#79da81ff',
};

export const fontSizes = { h1: 20, body: 14, caption: 10 };
export const lineHeights = {
  h1: 20,    // ← Figma 16은 실제 디바이스에서 잘림 발생 → 20으로 올려서 클리핑 방지
  body: 20,
  caption: 16,
};

export const fonts = {
  // Pretendard 폰트 패밀리명은 프로젝트에 설치/로드된 이름과 일치해야 함
  black: 'Pretendard-Black',
  bold: 'Pretendard-Bold',
  semibold: 'Pretendard-SemiBold',
  medium: 'Pretendard-Medium',
  regular: 'Pretendard-Regular',
};

export const sizes = {
  icon: 30,     // 하단 탭 아이콘 30x30
  circle: 250,  // 홈 중앙 원 250x250
};

// 모서리/섀도우 등 필요한 기본 토큰
export const radii = { round: 999, md: 12 };