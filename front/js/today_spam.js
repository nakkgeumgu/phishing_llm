import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system";

const getToday = () => {
    const now = new Date();
    const timestamp = now.toISOString().replace("T", " ").substring(0, 10);
    return timestamp;
};

export const reload = async () => {
  try {
    const today = getToday();
    console.log(today)
    const fileUri = FileSystem.documentDirectory + "log.json";

    const content = await FileSystem.readAsStringAsync(fileUri);
    const logs = JSON.parse(content);

    const todaySpam = logs.filter((item) => {
      const isToday = item.date.startsWith(today);
      const isSpam = Number(item.label) === 1; 
      return isToday && isSpam;
    });
    
    await AsyncStorage.setItem("spam", JSON.stringify(todaySpam));
    await AsyncStorage.setItem("spam_num", todaySpam.length.toString());
    await AsyncStorage.setItem("spam_date", today);
  } catch (e) {
    console.error("reload error:", e);
  }
};

export const spam_json = async () => {
  try {
    const json = await AsyncStorage.getItem("spam");
    return json ? JSON.parse(json) : [];
  } catch (e) {
    console.error("spam_json error:", e);
    return [];
  }
};

export const spam_num = async () => {
  try {
    const num = await AsyncStorage.getItem("spam_num");
    return num ? parseInt(num) : 0;
  } catch (e) {
    console.error("spam_num error:", e);
    return 0;
  }
};