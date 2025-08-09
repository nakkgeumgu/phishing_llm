import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Button,
  Modal,
  TouchableOpacity,
  Platform,
  Pressable,
  ScrollView,
} from "react-native";
import { Calendar } from "react-native-calendars";
import { Picker } from "@react-native-picker/picker";
import { read_log } from "../../js/spam_manage.js"; 

export default function App() {
  const [logData, setLogData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [choiceDate, setChoiceDate] = useState(new Date());
  const [filterMode, setFilterMode] = useState("all");
  const [sortOption, setSortOption] = useState("datetime");
  const [showCalendar, setShowCalendar] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const cvtParamDate = (date) => {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
};

  const selectedDateStr = cvtParamDate(choiceDate);

  useEffect(() => {
  setIsLoading(true); 
  read_log()
    .then((log) => {
      setLogData(log); 
      setIsLoading(false); 
    })
    .catch((err) => {
      console.error("에러 발생:", err);
      setIsLoading(false);
    });
}, []);

  const onChange = (selectedDate) => {
    const currentDate = new Date(selectedDate.dateString);
    setChoiceDate(currentDate);
    setFilterMode("date");
    setCurrentPage(1);
    setShowCalendar(false);
  };

  let filteredLogs =
    filterMode === "all"
      ? [...logData]
      : logData.filter((item) => item.date.startsWith(selectedDateStr));

  // 정렬: 시간순 (최신순)
  filteredLogs.sort((a, b) => new Date(b.date) - new Date(a.date));

  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage);
  const paginatedLogs = filteredLogs.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const truncate = (text) =>
    text.length > 15 ? text.slice(0, 15) + "…" : text;

  return (
    <View style={{ padding: 20, paddingTop: 50 }}>
      {/* 상단 컨트롤 */}
      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <Button title="날짜 선택" onPress={() => setShowCalendar(true)} />
        <View
          style={{
            borderWidth: 1,
            borderRadius: 5,
            overflow: "hidden",
            width: 180,
            height: Platform.OS === "android" ? 40 : undefined,
          }}
        >
          <Picker
            selectedValue={sortOption}
            onValueChange={(value) => setSortOption(value)}
            mode="dropdown"
          >
            <Picker.Item label="시간순 정렬" value="datetime" />
            <Picker.Item label="확률순 정렬" value="probability" />
          </Picker>
        </View>
      </View>

      {/* 필터 상태 */}
      <Text style={{ fontSize: 20, marginBottom: 10, textAlign: "center" }}>
        {filterMode === "all" ? "전체 로그" : `${selectedDateStr}의 로그`}
      </Text>

      {/* 전체 보기 버튼 */}
      {filterMode !== "all" && (
        <View style={{ alignItems: "center", marginBottom: 10 }}>
          <Button title="전체 보기" onPress={() => {
            setFilterMode("all");
            setCurrentPage(1);
          }} />
        </View>
      )}

      {/* 표 형태 로그 */}
      <View style={{ borderWidth: 1, borderColor: "#ccc" }}>
        <View style={{ flexDirection: "row", backgroundColor: "#eee", padding: 8 }}>
          <Text style={{ flex: 3, fontWeight: "bold", textAlign: "center" }}>시간</Text>
          <Text style={{ flex: 4, fontWeight: "bold", textAlign: "center" }}>내용</Text>
          <Text style={{ flex: 2, fontWeight: "bold", textAlign: "center" }}>확률</Text>
        </View>

        {isLoading ? (
          <Text style={{ textAlign: "center", padding: 10 }}>불러오는 중...</Text>
        ) : paginatedLogs.length === 0 ? (
          <Text style={{ textAlign: "center", padding: 10 }}>해당 로그가 없습니다.</Text>
        ) : (
          paginatedLogs.map((item, index) => (
            <Pressable
              key={index}
              onPress={() => {
                setSelectedItem(item);
                setModalVisible(true);
              }}
            >
              <View
                style={{
                  flexDirection: "row",
                  borderTopWidth: 1,
                  borderColor: "#eee",
                  paddingVertical: 6,
                  paddingHorizontal: 4,
                }}
              >
                <Text style={{ flex: 3, textAlign: "center" }}>{item.date}</Text>
                <Text style={{ flex: 4, textAlign: "center" }}>{truncate(item.content)}</Text>
                <Text style={{ flex: 2, textAlign: "center" }}>{truncate(item.Probability + "%")}</Text>
              </View>
            </Pressable>
          ))
        )}
      </View>

      {/* 페이지네이션 */}
      {filteredLogs.length > itemsPerPage && (
        <View style={{ flexDirection: "row", justifyContent: "center", marginTop: 15 }}>
          <Button
            title="이전"
            onPress={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          />
          <Text style={{ marginHorizontal: 10, fontSize: 16 }}>
            {currentPage} / {totalPages}
          </Text>
          <Button
            title="다음"
            onPress={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
          />
        </View>
      )}

      {/* 상세 모달 */}
      <Modal visible={modalVisible} transparent animationType="slide">
        <View
          style={{
            flex: 1,
            backgroundColor: "#00000088",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <View
            style={{
              backgroundColor: "white",
              padding: 20,
              borderRadius: 10,
              width: "85%",
            }}
          >
            <Text style={{ fontSize: 18, marginBottom: 10, fontWeight: "bold" }}>
              상세 로그
            </Text>
            {selectedItem && (
              <ScrollView>
                <Text style={{ marginBottom: 5 }}>📅 시간: {selectedItem.date}</Text>
                <Text style={{ marginBottom: 5 }}>🎯 확률: {selectedItem.Probability}%</Text>
                <Text style={{ marginBottom: 5 }}>📝 내용: {selectedItem.content}</Text>
              </ScrollView>
            )}
            <TouchableOpacity
              onPress={() => setModalVisible(false)}
              style={{
                marginTop: 15,
                padding: 10,
                backgroundColor: "#eee",
                borderRadius: 5,
                alignItems: "center",
              }}
            >
              <Text>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* 날짜 선택 모달 */}
      <Modal visible={showCalendar} transparent animationType="slide">
        <View
          style={{
            flex: 1,
            backgroundColor: "#000000aa",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <View
            style={{
              backgroundColor: "white",
              padding: 20,
              borderRadius: 10,
              width: "90%",
            }}
          >
            <Calendar
              current={selectedDateStr}
              maxDate={cvtParamDate(new Date())}
              onDayPress={onChange}
              monthFormat="yyyy년 MM월"
            />
            <TouchableOpacity
              onPress={() => setShowCalendar(false)}
              style={{
                marginTop: 10,
                padding: 10,
                backgroundColor: "#eee",
                alignItems: "center",
                borderRadius: 5,
              }}
            >
              <Text>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}