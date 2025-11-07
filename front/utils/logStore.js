// ✅ Expo SDK 54: 레거시 FS API 고정
import * as FileSystem from 'expo-file-system/legacy';

const FILE_PATH = FileSystem.documentDirectory + 'phishing_logs.json';

// 저장을 직렬화하기 위한 간단 큐 (동시 저장 레이스 방지)
let writeQueue = Promise.resolve();

// 내부: 전체 읽기
async function readAll() {
  try {
    const info = await FileSystem.getInfoAsync(FILE_PATH);
    if (!info.exists) return [];
    // 기본 UTF-8
    const txt = await FileSystem.readAsStringAsync(FILE_PATH);
    const parsed = JSON.parse(txt);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn('readAll error', e);
    return [];
  }
}

// 내부: 전체 쓰기
async function writeAll(arr) {
  try {
    const txt = JSON.stringify(arr);
    await FileSystem.writeAsStringAsync(FILE_PATH, txt);
  } catch (e) {
    console.warn('writeAll error', e);
  }
}

/**
 * 로그 추가
 * @param {{id:string, dateISO:string, content:string, label:0|1, prob:number|null, createdAt:number}} entry
 */
export async function appendLog(entry) {
  // 직렬화된 쓰기 보장
  writeQueue = writeQueue
    .then(async () => {
      const logs = await readAll();
      logs.push(entry);
      await writeAll(logs);
    })
    .catch((e) => console.warn('appendLog queue error', e));

  return writeQueue;
}

/** 전체 로그 가져오기 */
export async function getLogs() {
  return await readAll();
}

/** 전체 삭제(디버그용) */
export async function clearLogs() {
  try {
    await FileSystem.deleteAsync(FILE_PATH, { idempotent: true });
  } catch (e) {
    console.warn('clearLogs error', e);
  }
}

// 필요하면 경로를 디버깅용으로 export
export { FILE_PATH };
