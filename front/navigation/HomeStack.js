import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from '../screens/community/HomeScreen';
import DetailScreen from '../screens/community/DetailScreen';
import WriteScreen from '../screens/community/WriteScreen';

const Stack = createStackNavigator();

export default function HomeStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={HomeScreen} options={{ title: '커뮤니티' }} />
      <Stack.Screen name="Detail" component={DetailScreen} options={{ title: '글 상세' }} />
      <Stack.Screen name="Write" component={WriteScreen} options={{ title: '글 쓰기' }} />
    </Stack.Navigator>
  );
}
