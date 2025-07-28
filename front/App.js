// App.js
import React from 'react';
import  { useState } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {
  SafeAreaView,
  FlatList,
  TouchableOpacity,
  View,
  Text,
  TextInput,
  StyleSheet,
} from 'react-native';
import HomeScreen from './screens/community/HomeScreen';
import DetailScreen from './screens/community/DetailScreen';
import WriteScreen from './screens/community/WriteScreen';
import { PostProvider } from './store/PostContext';
import TabNavigator from './navigation/TabNavigator';
import styles from './styles';
import { StatusBar } from 'react-native';



const Stack = createNativeStackNavigator();

// export default function App() {
//   return (
//     <SafeAreaProvider>
//       <PostProvider>  
//         <NavigationContainer>
//         <Stack.Navigator initialRouteName="Home">
//           <Stack.Screen name="Home" component={HomeScreen} options={{ title: '홈' }} />
//           <Stack.Screen name="Detail" component={DetailScreen} options={{ title: '글 상세' }} />
//           <Stack.Screen name="Write" component={WriteScreen} options={{ title: '글 쓰기' }} />
//         </Stack.Navigator>
//         </NavigationContainer>      
//       </PostProvider>
//     </SafeAreaProvider>
//   );
// }

export default function App(){
  return(
    
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor="#fafafa" />
      <PostProvider>
        <NavigationContainer>
          <TabNavigator />
        </NavigationContainer>    
      </PostProvider>
    </SafeAreaProvider>
  );
}