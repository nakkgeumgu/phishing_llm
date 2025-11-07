import { StyleSheet } from 'react-native';
import { colors, fontSizes, lineHeights } from './tokens';
import { clamp, getCircleSize, getSubtitleMarginH } from './helpers';

// RN letterSpacing은 px 단위 → 퍼센트 변환 유틸
const ls = (fs, percent = -0.025) => fs * percent;

export const createHomeStyles = (opts = {}) => {
  const { insetTop = 0, width = 393 } = opts;
  const CIRCLE = getCircleSize(width);
  const mh = getSubtitleMarginH(width);
  const headerHeight = Math.round(clamp(width * 0.15, 50, 60));

  return StyleSheet.create({
    screen: { flex: 1, backgroundColor: colors.background },

    header: { height: headerHeight, alignItems: 'center', justifyContent: 'center' },
    headerTitle: {
      fontFamily: 'Pretendard-Black', fontSize: fontSizes.h1, lineHeight: lineHeights.h1,
      letterSpacing: ls(fontSizes.h1), color: colors.text, textAlign: 'center', includeFontPadding: false,
    },

    body: { flex: 1, paddingHorizontal: 20 },

    subtitle: {
      fontFamily: 'Pretendard-Regular', fontSize: fontSizes.body, lineHeight: lineHeights.body,
      letterSpacing: ls(fontSizes.body), color: colors.textMuted, marginTop: 10, marginHorizontal: mh,
      textAlign: 'left', alignSelf: 'flex-start',
    },

    visualWrap: { alignItems: 'center', justifyContent: 'center', marginTop: 32, marginBottom: 18 },

    // 기본 원: 테두리/그림자 없음
    circle: {
      width: CIRCLE, height: CIRCLE, borderRadius: CIRCLE / 2, backgroundColor: colors.circle,
      alignItems: 'center', justifyContent: 'center', overflow: 'visible',
    },
    // 토글 ON: 테두리 + 그림자
    circleOutlineOn: {
      borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.35)',
      shadowColor: colors.accentGreen, shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.25, shadowRadius: 37.5,
      elevation: 18,
    },
    // 토글 OFF: 완전 제거
    circleOutlineOff: { borderWidth: 0, borderColor: 'transparent', shadowOpacity: 0, shadowRadius: 0, elevation: 0 },

    innerHighlight: {
      position: 'absolute', top: 0, left: 0, right: 0, height: CIRCLE * 0.55,
      borderTopLeftRadius: CIRCLE / 2, borderTopRightRadius: CIRCLE / 2,
      backgroundColor: 'rgba(255,255,255,0.25)', opacity: 0.25,
    },
    icon: { width: CIRCLE * 0.54, height: CIRCLE * 0.54, resizeMode: 'contain' },

    // 상태 문구: 켜짐/꺼짐 스타일 분리
    statusOn: {
      fontFamily: 'Pretendard-Black', fontSize: fontSizes.h1, lineHeight: lineHeights.h1,
      letterSpacing: ls(fontSizes.h1), color: colors.text, textAlign: 'center', marginTop: 40, includeFontPadding: false,
    },
    statusOff: {
      fontFamily: 'Pretendard-Black', fontSize: fontSizes.h1, lineHeight: lineHeights.h1,
      letterSpacing: ls(fontSizes.h1), color: colors.textMuted, textAlign: 'center', marginTop: 40, includeFontPadding: false,
    },

    recent: {
      fontFamily: 'Pretendard-Regular', fontSize: fontSizes.body, lineHeight: lineHeights.body,
      letterSpacing: ls(fontSizes.body), color: colors.textMuted, marginTop: 20, marginHorizontal: mh,
      textAlign: 'left', alignSelf: 'flex-start',
    },
  });
};


