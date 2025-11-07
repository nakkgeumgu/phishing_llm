import React from 'react';
import { View, Text, Image, StatusBar, Pressable, Animated, useWindowDimensions } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { createHomeStyles } from '../../design/homeStyles';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const styles = React.useMemo(() => createHomeStyles({ insetTop: insets.top, width }), [insets.top, width]);

  const [outlineOn, setOutlineOn] = React.useState(true);
  const toggleOutline = () => setOutlineOn(v => !v);

  // scale 애니메이션 값
  const scale = React.useRef(new Animated.Value(1)).current;
  const bumpIn = () => {
    Animated.spring(scale, { toValue: 1.06, useNativeDriver: true, friction: 5, tension: 120 }).start();
  };
  const bumpOut = () => {
    Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 6, tension: 140 }).start();
  };

  const statusText = outlineOn ? '피싱 메시지 탐지 중' : '탐지 일시중지됨';

  return (
    <SafeAreaView style={styles.screen} edges={['top']}>
      <StatusBar barStyle="light-content" backgroundColor="#181A20" />

      <View style={styles.header}>
        <Text style={styles.headerTitle}>낚시 금지 구역</Text>
      </View>

      <View style={styles.body}>
        <Text style={styles.subtitle}>
          실시간 탐지 기능은 백그라운드에서 메시지를 분석해 피싱 여부를 판단합니다.
        </Text>

        <View style={styles.visualWrap}>
          <Pressable
            onPress={toggleOutline}     // 토글
            onPressIn={bumpIn}          // 커졌다가
            onPressOut={bumpOut}        // 돌아오기
            accessibilityRole="button"
            accessibilityLabel={outlineOn ? '아웃라인 끄기' : '아웃라인 켜기'}
          >
            <Animated.View
              style={[
                styles.circle,
                outlineOn ? styles.circleOutlineOn : styles.circleOutlineOff,
                { transform: [{ scale }] },
              ]}
            >
              {outlineOn && <View pointerEvents="none" style={styles.innerHighlight} />}
              <Image source={require('../../assets/fish.png')} style={styles.icon} />
            </Animated.View>
          </Pressable>
        </View>

        <Text style={outlineOn ? styles.statusOn : styles.statusOff}>{statusText}</Text>
        <Text style={styles.recent}>최근 탐지된 피싱 메시지 없음</Text>
      </View>
    </SafeAreaView>
  );
}
