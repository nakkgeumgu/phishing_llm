import { Platform } from 'react-native';
import { colors, sizes, fontSizes } from './tokens';
import { Dimensions } from 'react-native';

// RN의 letterSpacing은 'pt' 단위이므로 퍼센트를 pt로 변환
export const ls = (fontSize, percent = -0.025) => fontSize * percent; // -2.5%

// clamp 유틸
export const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

// 현재 화면 너비/높이
export const win = () => Dimensions.get('window');

// 홈 원형 크기: 화면에 맞춰 180~250 사이에서 자동
export const getCircleSize = (w = win().width) => Math.round(clamp(w * 0.62, 180, 250));

// 설명문구 좌우 마진: 화면에 맞춰 20~57 사이에서 자동
export const getSubtitleMarginH = (w = win().width) => Math.round(clamp(w * 0.145, 20, 57));

// 하단 탭/텍스트 등 공통 타이포 유틸
export const type = {
  // Pretendard Black 20 / lh 16 / -2.5%
  titleBlack20: {
    fontFamily: 'Pretendard-Black',
    fontSize: fontSizes.h1,
    lineHeight: 16,
    letterSpacing: ls(fontSizes.h1),
    color: colors.text,
  },
  // Pretendard Regular 14 / lh 20 / -2.5%
  body14: {
    fontFamily: 'Pretendard-Regular',
    fontSize: fontSizes.body,
    lineHeight: 20,
    letterSpacing: ls(fontSizes.body),
    color: colors.textMuted,
  },
  // Pretendard Medium 10 / lh 16 / -2.5% (탭 라벨)
  caption10: {
    fontFamily: 'Pretendard-Medium',
    fontSize: fontSizes.caption,
    lineHeight: 16,
    letterSpacing: ls(fontSizes.caption),
    color: colors.text,
  },
};

// 안드로이드 섀도우 보정(elevation), iOS는 shadow* 사용
export const bigGlowShadow = {
  // Figma: 바깥쪽 그림자 (흐림 75, 색상 #07540C, 불투명도 25%)
  // RN 매핑: shadowRadius ~ blur, opacity ~ 0.25
  shadowColor: colors.accentGreen,
  shadowOffset: { width: 0, height: 20 },
  shadowOpacity: 0.25,
  shadowRadius: 37.5, // 75의 절반 정도로 매핑(시각 보정)
  ...(Platform.OS === 'android' ? { elevation: 18 } : null),
};

// "안쪽 그림자(흐림 2, 흰색 25%)"는 RN 기본만으로는 불가.
// 아래는 상단-좌측 하이라이트를 주는 오버레이 스타일(Gradient나 반투명 View를 함께 사용).
export const innerHighlightOverlay = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  height: sizes.circle * 0.55,
  borderTopLeftRadius: sizes.circle / 2,
  borderTopRightRadius: sizes.circle / 2,
  backgroundColor: 'rgba(255,255,255,0.25)',
  opacity: 0.25,
};

// 원형 사이즈/아이콘 정비
export const circleMetrics = {
  size: sizes.circle,
  radius: sizes.circle / 2,
  iconPercent: 0.54, // 아이콘 54%
};