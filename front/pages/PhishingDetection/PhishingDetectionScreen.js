import React from 'react';
import {
  View, Text, TextInput, Pressable, Animated, StatusBar,
  useWindowDimensions, Alert, Keyboard, TouchableWithoutFeedback, Platform
} from 'react-native';
import { InputAccessoryView } from 'react-native'; // iOS 전용
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import LottieView from 'lottie-react-native';
import * as Clipboard from 'expo-clipboard';

import { appendLog } from '../../utils/logStore';
import 'react-native-get-random-values'; // UUID 쓰면 필요
import { v4 as uuidv4 } from 'uuid';     // 쓸 거면 설치: npm i uuid  (선택)

import { colors } from '../../design/tokens';
import { createPhishingStyles } from '../../design/phishingStyles';
import { createHomeStyles } from '../../design/homeStyles'; // 헤더/배경 재사용

const baseURL = __DEV__ ? 'http://15.134.160.211:5000' : 'http://15.134.160.211:5000';

export default function PhishingDetectionScreen() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();

  const home = React.useMemo(() => createHomeStyles({ insetTop: insets.top, width }), [insets.top, width]);
  const styles = React.useMemo(() => createPhishingStyles({ width }), [width]);

  const [message, setMessage] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const [result, setResult] = React.useState(null); // {label, prob}
  const [errorText, setErrorText] = React.useState('');

  // 입력 커서/선택 범위 (붙여넣기 위치용)
  const [selection, setSelection] = React.useState({ start: 0, end: 0 });

  // iOS InputAccessoryView id
  const inputAccessoryViewID = 'phishAccessory';

  // Android용 키보드 표시/높이
  const [kbVisible, setKbVisible] = React.useState(false);
  const [kbH, setKbH] = React.useState(0);
  React.useEffect(() => {
    const showSub = Keyboard.addListener('keyboardDidShow', (e) => {
      setKbVisible(true);
      setKbH(e.endCoordinates?.height ?? 0);
    });
    const hideSub = Keyboard.addListener('keyboardDidHide', () => {
      setKbVisible(false);
      setKbH(0);
    });
    return () => { showSub.remove(); hideSub.remove(); };
  }, []);

  // 버튼 터치 애니메이션
  const scale = React.useRef(new Animated.Value(1)).current;
  const pressIn = () =>
    Animated.spring(scale, { toValue: 0.98, useNativeDriver: true, friction: 6, tension: 160 }).start();
  const pressOut = () =>
    Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 6, tension: 140 }).start();

  const handleDetection = async () => {
    if (isLoading) return;
    Keyboard.dismiss(); // 버튼 누르면 키보드 닫기
    if (message.trim() === '') {
      Alert.alert('입력 오류', '메시지를 입력해주세요.');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setErrorText('');

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);

    try {
      const payload = { text: message, date: new Date().toISOString() };
      const response = await fetch(`${baseURL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        const t = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status} ${t || ''}`.trim());
      }

      const data = await response.json();
      const label = data.label ?? 'unknown';
      const prob = typeof data.phishing_prob === 'number' ? data.phishing_prob : NaN;
      setResult({ label, prob: Number.isFinite(prob) ? prob : null });
       // ✅ 파일에 영구 저장
        const now = new Date();
         const localYMD = new Date(
           now.getFullYear(), now.getMonth(), now.getDate()
         ).toLocaleDateString('en-CA'); // ✅ YYYY-MM-DD 로컬 기준
         const entry = {
           id: String(now.getTime()),
           dateISO: localYMD,
         createdAt: now.getTime(),
         content: message,
         label,
         prob: Number.isFinite(prob) ? prob : null,
       };
       await appendLog(entry);
    } catch (err) {
      console.error('API 요청 실패:', err);
      setErrorText(err?.name === 'AbortError' ? '시간 초과: 네트워크를 확인해주세요.' : '서버 요청 실패');
    } finally {
      clearTimeout(timer);
      setIsLoading(false);
    }
  };

  const isDisabled = isLoading || message.trim().length === 0;

  // 붙여넣기
  const pasteFromClipboard = async () => {
    const clip = await Clipboard.getStringAsync();
    if (!clip) return;
    setMessage((prev) => {
      const { start, end } = selection || { start: prev.length, end: prev.length };
      const left = prev.slice(0, start);
      const right = prev.slice(end);
      const next = left + clip + right;
      // 커서를 붙여넣은 뒤로 이동
      requestAnimationFrame(() => {
        const pos = (left + clip).length;
        setSelection({ start: pos, end: pos });
      });
      return next;
    });
  };

  // 결과 문구
  const renderResult = () => {
    if (errorText) {
      return (
        <View style={styles.resultWrap}>
          <Text style={[styles.resultLine, styles.resultMuted]}>❌ {errorText}</Text>
        </View>
      );
    }
    if (!result) return null;

    // 라벨: 1=피싱, 0=피싱 아님 (서버 그대로)
  const isPhishing = Number(result.label) === 1;
  // 확률: 서버가 준 phishing_prob 그대로 사용 (반전/정규화 X)
  const p = Number.isFinite(result.prob) ? result.prob : null; // 0..1 가정
  const probPct = p == null ? null : (p * 100).toFixed(2);
  // 요구사항: "피싱일 가능성만 있으면" => 라벨이 1이거나, 확률이 0보다 크면
  const hasPossibility = isPhishing;

    return (
      <View style={styles.resultWrap}>
     {/* 1줄 요약: 가능성만 있으면 고정 문구 */}
     <Text style={styles.resultLine}>
       {hasPossibility ? '피싱일 가능성이 있어요.' : '피싱 가능성이 거의 없어요.'}
     </Text>
     {/* 2줄: 서버 확률 그대로 표시 */}
     <Text style={[styles.resultLine, styles.resultMuted]}>
       AI 검사결과 피싱일 확률은{' '}
       <Text style={styles.resultEmphasisRed}>{probPct ?? '-' }%</Text> 입니다.
     </Text>
      </View>
    );
  };

  return (
    // 바깥 아무데나 탭하면 키보드 닫힘
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
      <SafeAreaView style={home.screen} edges={['top']}>
        <StatusBar barStyle="light-content" backgroundColor={colors.background} />

        {/* 헤더/설명: 홈과 동일 */}
        <View style={styles.header}>
          <Text
            style={styles.headerTitle}
            numberOfLines={1}
            adjustsFontSizeToFit
            minimumFontScale={0.85}
            ellipsizeMode="tail"
          >
            피싱 판별
          </Text>
        </View>

        <View style={styles.body}>
          <Text style={styles.subtitle}>
            의심스러운 메시지를 입력하면 AI가 피싱 메시지인지 판별해줘요.
          </Text>

          {/* 입력 */}
          <View style={styles.inputWrap}>
            <TextInput
              style={styles.input}
              placeholder="여기에 내용을 입력해주세요!"
              placeholderTextColor={styles.placeholder?.color || '#8A909D'}
              multiline
              submitBehavior="newline"         // 엔터=줄바꿈, 포커스 유지
              returnKeyType="default"
              scrollEnabled
              showsVerticalScrollIndicator
              underlineColorAndroid="transparent"
              value={message}
              onChangeText={setMessage}
              onSelectionChange={(e) => setSelection(e.nativeEvent.selection)}
              selection={selection}
              {...(Platform.OS === 'ios' ? { inputAccessoryViewID } : {})}
            />

            {/* X(전체 지우기) */}
            {message.length > 0 && (
              <Pressable
                onPress={() => setMessage('')}
                style={styles.clearBtn}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                accessibilityRole="button"
                accessibilityLabel="입력 내용 지우기"
              >
                <Text style={styles.clearIcon}>×</Text>
              </Pressable>
            )}

            {/* iOS 전용: 키보드 위 액세서리 바 */}
            {Platform.OS === 'ios' && (
              <InputAccessoryView nativeID={inputAccessoryViewID}>
                <View style={styles.accessoryBar}>
                  <Pressable onPress={pasteFromClipboard} style={styles.accessoryBtn}>
                    <Text style={styles.accessoryText}>붙여넣기</Text>
                  </Pressable>
                  <Pressable onPress={() => setMessage('')} style={styles.accessoryBtn}>
                    <Text style={styles.accessoryText}>지우기</Text>
                  </Pressable>
                </View>
              </InputAccessoryView>
            )}
          </View>

          {/* 검사 버튼 */}
          <Pressable onPress={handleDetection} onPressIn={pressIn} onPressOut={pressOut} disabled={isDisabled}>
            <Animated.View style={[styles.button, isDisabled && styles.buttonDisabled, { transform: [{ scale }] }]}>
              <Text style={styles.buttonText}>검사하기</Text>
            </Animated.View>
          </Pressable>

          {/* 로딩/결과 */}
          {isLoading ? (
            <View style={{ alignItems: 'center', marginTop: 30 }}>
              <LottieView
                source={require('../../assets/gradient_loader_01.json')}
                autoPlay
                loop
                style={{ width: 240, height: 240 }}
              />
              <Text style={[styles.resultLine, styles.resultMuted]}>검사 중입니다...</Text>
            </View>
          ) : (
            renderResult()
          )}
        </View>

        {/* Android 대체 액세서리 바: 키보드 위에 고정 */}
        {Platform.OS === 'android' && kbVisible && (
          <View style={[styles.accessoryBar, { position: 'absolute', left: 0, right: 0, bottom: kbH }]}>
            <Pressable onPress={pasteFromClipboard} style={styles.accessoryBtn}>
              <Text style={styles.accessoryText}>붙여넣기</Text>
            </Pressable>
            <Pressable onPress={() => setMessage('')} style={styles.accessoryBtn}>
              <Text style={styles.accessoryText}>지우기</Text>
            </Pressable>
          </View>
        )}
      </SafeAreaView>
    </TouchableWithoutFeedback>
  );
}
