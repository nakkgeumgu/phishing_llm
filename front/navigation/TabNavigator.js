import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeStack from './HomeStack';

const Tab = createBottomTabNavigator();

export default function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
      }}
    >
      <Tab.Screen name="AI 피싱 판별" component={HomeStack}/>
      <Tab.Screen name="로그" component={HomeStack}/>
      <Tab.Screen name="홈" component={HomeStack} />
      <Tab.Screen name="커뮤니티" component={HomeStack} />
      <Tab.Screen name="test" component={HomeStack} />
    </Tab.Navigator>
  );
}