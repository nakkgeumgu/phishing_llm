import React, { useContext } from 'react';
import { SafeAreaView, FlatList, TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { PostContext } from '../../store/PostContext';
import styles from '../../styles';

function HomeScreen({ navigation }) { 
  const {posts} = useContext(PostContext);

    const MAX_LENGTH = 80;

    const renderItem = ({ item }) => {
      const previewContent =
        item.content.length > MAX_LENGTH
          ? item.content.substring(0, MAX_LENGTH) + '...'
          : item.content;

      return (
        <TouchableOpacity
          style={cardStyles.card}
          onPress={() => navigation.navigate('Detail', { post: item })}
        >
          <View style={cardStyles.header}>
            <View style={cardStyles.avatarPlaceholder} />
            <Text style={cardStyles.username}>{item.author}</Text>
          </View>

          <Text style={cardStyles.title}>{item.title}</Text>
          <Text style={cardStyles.content}>{previewContent}</Text>
        </TouchableOpacity>
      );
    };

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        data={[...posts].sort((a, b) => b.id - a.id)}
        keyExtractor={item => item.id?.toString() ?? Math.random().toString()}
        renderItem={renderItem}
        contentContainerStyle={styles.list}
      />
      <TouchableOpacity
        style={styles.fab}
        onPress={() => navigation.navigate('Write')}
      >
        <Text style={styles.fabIcon}>＋</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const cardStyles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 6,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  avatarPlaceholder: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#ddd',
    marginRight: 10,
  },
  username: {
    fontWeight: '600',
    fontSize: 15,
  },
  title: {
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 4,
  },
  content: {
    fontSize: 14,
    color: '#444',
  },
});

export default HomeScreen;