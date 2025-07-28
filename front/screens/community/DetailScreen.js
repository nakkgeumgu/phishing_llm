import React, { useContext } from 'react';
import { SafeAreaView, View, Text, StyleSheet } from 'react-native';
import styles from '../../styles';

function DetailScreen({ route }) {
  const { post } = route.params;
  return (
      <SafeAreaView style={detailStyles.container}>
        <View style={detailStyles.card}>
          {/* 상단: 프로필 + 작성자 */}
          <View style={detailStyles.header}>
            <View style={detailStyles.avatarPlaceholder} />
            <Text style={detailStyles.username}>{post.author}</Text>
          </View>

          {/* 제목 */}
          <Text style={detailStyles.title}>{post.title}</Text>

          {/* 본문 */}
          <Text style={detailStyles.content}>{post.content}</Text>
        </View>
      </SafeAreaView>
    );
}

const detailStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fafafa',
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 6,
    elevation: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  avatarPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#ddd',
    marginRight: 12,
  },
  username: {
    fontSize: 16,
    fontWeight: '600',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#222',
  },
  content: {
    fontSize: 16,
    lineHeight: 24,
    color: '#444',
  },
});

export default DetailScreen;