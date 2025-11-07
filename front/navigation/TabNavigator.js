import React from 'react';
import { Image } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

// screens
import HomeScreen from '../pages/Home/HomeScreen';
import PhishingDetectionScreen from '../pages/PhishingDetection/PhishingDetectionScreen';
import PhishingLogScreen from '../pages/Log/PhishingLogScreen';

// design
import { tabBarStyles } from '../design/navStyles';

const Tab = createBottomTabNavigator();

export default function TabNavigator() {
  return (
    <Tab.Navigator screenOptions={tabBarStyles.screenOptions}>
      {/* 홈 */}
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          title: '홈',
          tabBarIcon: ({ focused, color, size }) => (
            <Image
              source={
                focused
                  ? require('../assets/icons/home_active.png')
                  : require('../assets/icons/home.png')
              }
              style={{ width: size ?? tabBarStyles.iconSize, height: size ?? tabBarStyles.iconSize, tintColor: color, opacity: focused ? 1 : 0.9 }}
              resizeMode="contain"
            />
          ),
        }}
      />

      {/* 피싱 판별 */}
      <Tab.Screen
        name="PhishingDetect"
        component={PhishingDetectionScreen}
        options={{
          title: '피싱 판별',
          tabBarIcon: ({ focused, color, size }) => (
            <Image
              source={
                focused
                  ? require('../assets/icons/doc_active.png')
                  : require('../assets/icons/doc.png')
              }
              style={{ width: size ?? tabBarStyles.iconSize, height: size ?? tabBarStyles.iconSize, tintColor: color, opacity: focused ? 1 : 0.9 }}
              resizeMode="contain"
            />
          ),
        }}
      />
      {/* 피싱 로그 */}
      <Tab.Screen
        name="PhishingLog"
        component={PhishingLogScreen}
        options={{
          title: '피싱 로그',
          tabBarIcon: ({ focused, color, size }) => (
            <Image
              source={focused ? require('../assets/icons/search_active.png') : require('../assets/icons/search.png')}
              style={{ width: size ?? tabBarStyles.iconSize, height: size ?? tabBarStyles.iconSize, tintColor: color }}
            />
          ),
        }}
      />
    </Tab.Navigator>
  );
}
