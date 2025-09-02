import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import CommunityStack from './HomeStack';
import LogScreen from '../screens/log/logscreen';
import PanbyeolScreen from '../screens/panbyeol/App';
import MainScreen from '../screens/main/MainScreen';

const Tab = createBottomTabNavigator();

export default function TabNavigator() {
  return (
    <Tab.Navigator screenOptions={{ headerShown: false }}>
      <Tab.Screen name="AI 피싱 판별" component={PanbyeolScreen} />
      <Tab.Screen name="홈" component={MainScreen} />
      <Tab.Screen name="로그" component={LogScreen} />
      <Tab.Screen name="커뮤니티" component={CommunityStack} />
    </Tab.Navigator>
  );
}
