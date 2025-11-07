import { StyleSheet } from 'react-native';
import { colors, fontSizes, lineHeights } from './tokens';
import { clamp, getSubtitleMarginH } from './helpers';

// letterSpacing util
const ls = (fs, p = -0.025) => fs * p;

export const createPhishingStyles = (opts = {}) => {
  const { width = 393 } = opts;

  const mh = getSubtitleMarginH(width);
  const headerHeight = Math.round(clamp(width * 0.15, 50, 60)); // 홈과 동일 로직

  return StyleSheet.create({
    // 상단(홈과 동일 구조)
    header: { height: headerHeight, alignItems: 'center', justifyContent: 'center' },
    headerTitle: {
      fontFamily: 'Pretendard-Black',
      fontSize: fontSizes.h1,
      lineHeight: lineHeights.h1,
      letterSpacing: ls(fontSizes.h1),
      color: colors.text,
      textAlign: 'center',
      includeFontPadding: false,
    },
    body: { flex: 1, paddingHorizontal: 20 },

    // 설명문구 (홈과 동일: 좌상단, 좌우 20~57)
    subtitle: {
      fontFamily: 'Pretendard-Regular',
      fontSize: fontSizes.body,
      lineHeight: lineHeights.body,
      letterSpacing: ls(fontSizes.body),
      color: colors.textMuted,
      marginTop: 10,
      marginHorizontal: mh,
      textAlign: 'left',
      alignSelf: 'flex-start',
    },

    // 입력 영역
    inputWrap: {
      marginTop: 24,
      marginHorizontal: 20,
      backgroundColor: '#2A2E37',
      borderRadius: 18,
      borderWidth: 1,
      borderColor: '#3A3F49',
      padding: 16,
      height: 210,
      position: 'relative',
    },
    input: {
      height : "100%",
      fontFamily: 'Pretendard-Regular',
      fontSize: 16,
      lineHeight: 22,
      color: '#E7EAF0',
      textAlignVertical: 'top',
      paddingRight: 40,
    },
    clearBtn: {
        position: 'absolute',
        top : 10,
        right : 10,
        width: 28,
        height: 28,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
    },
    clearIcon:{
        fontFamily: 'Pretendard-SemiBold',
        fontSize: 18,
        color : '#8A909D',
    },

    placeholder: { color: '#8A909D' },

    // 버튼
    button: {
      marginTop: 28,
      marginHorizontal: 20,
      height: 56,
      borderRadius: 18,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#2563EB', // 블루 버튼
    },
    buttonText: {
      fontFamily: 'Pretendard-SemiBold',
      fontSize: 16,
      color: '#FFFFFF',
      letterSpacing: ls(16, -0.01),
    },
    buttonDisabled: {
      opacity: 0.5,
    },

    // 결과
    resultWrap: {
      marginTop: 18,
      marginHorizontal: 20,
    },
    resultLine: {
      fontFamily: 'Pretendard-Regular',
      fontSize: 14,
      lineHeight: 20,
      color: colors.text,
    },
    // createPhishingStyles() 내부 StyleSheet에 아래 3개를 포함하세요.
    accessoryBar: {
    backgroundColor: '#21232B',
    borderTopWidth: 1,
    borderTopColor: '#2E323B',
    paddingHorizontal: 12,
    paddingVertical: 8,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
    },
    accessoryBtn: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: '#2F3440',
    },
    accessoryText: {
    fontFamily: 'Pretendard-Medium',
    fontSize: 13,
    color: '#EDEFF4',
    },
    resultEmphasisRed: { color: '#EF4444', fontFamily: 'Pretendard-Bold' },
    resultMuted: { color: colors.textMuted },
  });
};
