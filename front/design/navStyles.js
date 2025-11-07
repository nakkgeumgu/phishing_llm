import { Platform } from 'react-native';
import { colors } from './tokens';
import { type } from './helpers';

export const tabBarStyles = {
  screenOptions: {
    headerShown: false,
    tabBarStyle: {
      backgroundColor: colors.tabBar, // 21232B
      height: Platform.select({ ios: 88, android: 72 }), // Figma 393x89 근사치
      borderTopWidth: 0,
      paddingTop: 6,
      paddingBottom: Platform.select({ ios: 22, android: 12 }),
    },
    tabBarActiveTintColor: colors.text,
    tabBarInactiveTintColor: colors.text,
    tabBarLabelStyle: {
      ...type.caption10, // Pretendard Medium 10 / lh 16 / -2.5%
      // React Navigation은 lineHeight 적용이 제한적일 수 있어 약간의 padding 보조
      paddingTop: 2,
    },
  },
  iconSize: 30, // 아이콘 30x30 (페이지에서 적용)
};
