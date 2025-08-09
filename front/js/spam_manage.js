import * as FileSystem from "expo-file-system";

const LOG_FILE = FileSystem.documentDirectory + "log.json";

export async function save_log(content, prob, label) {
  const now = new Date();
  const timestamp = now.toISOString().replace("T", " ").substring(0, 19); // YYYY-MM-DD HH:MM:SS
  const probability = Math.round(prob * 100); // 백분율로 변환

  const newLog = {
    date: timestamp,
    content,
    Probability: String(probability),
    label,
  };

  try {
    let existingLogs = [];
    const fileInfo = await FileSystem.getInfoAsync(LOG_FILE);
    if (fileInfo.exists) {
      const contentRaw = await FileSystem.readAsStringAsync(LOG_FILE);
      existingLogs = JSON.parse(contentRaw);
    }
    existingLogs.push(newLog);
    await FileSystem.writeAsStringAsync(LOG_FILE, JSON.stringify(existingLogs, null, 2));
  } catch (err) {
    console.error("로그 저장 실패:", err);
  }
}

export async function read_log() {
  try {
    const fileInfo = await FileSystem.getInfoAsync(LOG_FILE);
    if (!fileInfo.exists) {
      return [];
    }

    const contentRaw = await FileSystem.readAsStringAsync(LOG_FILE);
    return JSON.parse(contentRaw);
  } catch (err) {
    console.error("로그 읽기 실패:", err);
    return [];
  }
}

/* log 생성 테스트용 */
export async function send_llm(content) {
  try {
    const response = await fetch("http://192.168.1.101:5556/logs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content }),
    });

    if (!response.ok) {
      throw new Error(`서버 응답 오류: ${response.status}`);
    }

    const result = await response.json();

    if (!Array.isArray(result) || result.length === 0) {
      throw new Error("비정상 응답 형식");
    }

    const { phishing_prob, label } = result[0];
    await save_log(content, phishing_prob, label);

    return { phishing_prob, label };
  } catch (error) {
    console.error("send_llm 오류:", error);
    return null;
  }
}