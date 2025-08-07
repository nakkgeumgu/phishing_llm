import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { StyleSheet, TextInput, View, Button, Text, Alert } from 'react-native';
import LottieView from 'lottie-react-native';

export default function App() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState('');

  const handleDetection = async () => {
    if (message.trim() === '') {
      Alert.alert('입력 오류', '메시지를 입력해주세요.');
      return;
    }

    setIsLoading(true);
    setResult('');

    try {
      const response = await fetch('http://<BACKEND_URL>/api/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
      });

      const data = await response.json();

      // 서버 응답에서 result 문자열 추출ㄹ
      setResult(data.result || '결과를 받아오지 못했습니다.');
    } catch (error) {
      console.error('API 요청 실패:', error);
      setResult('❌ 서버 요청 실패');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="메시지를 입력하세요"
        value={message}
        onChangeText={setMessage}
      />

      <Button
        title="확인"
        color="#999"
        onPress={handleDetection}
      />

      {isLoading && (
        <View style={{ alignItems: 'center', marginTop: 30 }}>
          <LottieView
            source={require('./assets/gradient loader 01.json')}
            autoPlay
            loop
            style={{ width: 300, height: 300 }}
          />
          <Text style={styles.loadingText}>검사 중입니다...</Text>
        </View>
      )}

      {!isLoading && result !== '' && (
        <Text style={styles.resultText}>{result}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000ff',
    alignItems: 'center',
    justifyContent: 'flex-start',
  },
  input: {
    borderBottomWidth: 1.5,
    borderColor: '#01080eff',
    padding: 12,
    fontSize: 16,
    backgroundColor: '#b0b7baff',
    color: '#484a4aff',
    marginTop: 100,
    borderRadius: 17,
    width: '80%',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#fff',
    fontWeight: 'bold',
  },
  resultText: {
    marginTop: 30,
    fontSize: 18,
    color: '#ffffff',
    fontWeight: 'bold',
  },
});
