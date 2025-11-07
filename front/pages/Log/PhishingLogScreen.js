// pages/Log/PhishingLogScreen.js
import React, { useMemo, useCallback, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import {
  View, Text, Pressable, FlatList, useWindowDimensions, Platform,
  Image, Modal, ScrollView
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { colors, fontSizes, lineHeights } from '../../design/tokens';
import { getSubtitleMarginH } from '../../design/helpers';
import { createHomeStyles } from '../../design/homeStyles';
import { getLogs } from '../../utils/logStore';

const ls = (fs, p = -0.025) => fs * p;

// 로컬 YYYY-MM-DD
const ymd = (d) => {
  if (!(d instanceof Date)) return null;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
};

// 해당 월의 날짜 배열
const getMonthDays = (base) => {
  const y = base.getFullYear();
  const m = base.getMonth();
  const last = new Date(y, m + 1, 0).getDate();
  const days = [];
  for (let d = 1; d <= last; d++) days.push(new Date(y, m, d));
  return days;
};

export default function PhishingLogScreen() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();

  // 공통 헤더/배경
  const home = useMemo(() => createHomeStyles({ insetTop: insets.top, width }), [insets.top, width]);

  // 화면 전용 스타일
  const s = useMemo(() => {
    const mh = getSubtitleMarginH(width);
    const headerH = Math.round(Math.min(Math.max(width * 0.15, 50), 60));
    return {
      header: { height: headerH, alignItems: 'center', justifyContent: 'center' },
      headerTitle: {
        fontFamily: 'Pretendard-Black',
        fontSize: fontSizes.h1,
        lineHeight: lineHeights.h1,
        letterSpacing: ls(fontSizes.h1),
        color: colors.text,
        textAlign: 'center',
        includeFontPadding: false,
      },
      subtitle: {
        fontFamily: 'Pretendard-Regular',
        fontSize: fontSizes.body,
        lineHeight: lineHeights.body,
        letterSpacing: ls(fontSizes.body),
        color: colors.textMuted,
        marginTop: 10,
        marginHorizontal: mh,
        textAlign: 'left',
        alignSelf: 'flex-start',
        paddingHorizontal: 20, // 홈/판별과 좌우 여백 일치
      },

      // 상단 컨트롤
      controls: { marginTop: 16, paddingHorizontal: 20, flexDirection: 'row', alignItems: 'center', gap: 10 },
      dateIconBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: '#2F3440', alignItems: 'center', justifyContent: 'center' },
      dateIcon: { width: 18, height: 18, tintColor: '#EDEDED' },
      dateText: { color: '#EDEDED', fontFamily: 'Pretendard-Medium', fontSize: 13, marginRight: 8 },
      segWrap: { flexDirection: 'row', backgroundColor: '#2F3440', borderRadius: 24, padding: 4, gap: 6, marginLeft: 'auto' },
      segBtn: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 18 },
      segText: { fontFamily: 'Pretendard-Medium', fontSize: 12, color: '#BFC5D2' },
      segActive: { backgroundColor: '#252932', borderWidth: 1, borderColor: '#3A3F49' },
      segActiveText: { color: '#E7EAF0' },

      // 리스트
      list: { paddingTop: 16, paddingHorizontal: 14, paddingBottom: 40 },
      card: { backgroundColor: '#252932', borderRadius: 16, marginBottom: 18, overflow: 'hidden' },
      cardHeader: { backgroundColor: '#2F3440', paddingVertical: 10, paddingHorizontal: 14, flexDirection: 'row', justifyContent: 'space-between' },
      headText: { color: '#BFC5D2', fontFamily: 'Pretendard-Medium', fontSize: 12, letterSpacing: ls(12), lineHeight: 16 },
      row: { paddingHorizontal: 14, paddingVertical: 14, flexDirection: 'row', gap: 14 },
      dateCol: { width: 96 },
      dateTextCol: { color: '#D5D9E2', fontFamily: 'Pretendard-Regular', fontSize: 13, lineHeight: 20 },
      contentCol: { flex: 1 },
      contentText: { color: '#D5D9E2', fontFamily: 'Pretendard-Regular', fontSize: 13, lineHeight: 20 },
      probCol: { width: 64, alignItems: 'flex-end' },
      probText: { color: '#EF4444', fontFamily: 'Pretendard-SemiBold', fontSize: 13, lineHeight: 20 },

      empty: { alignItems: 'center', marginTop: 40 },
      emptyText: { color: '#9EA3B2', fontFamily: 'Pretendard-Regular', fontSize: 13, lineHeight: 20 },

      // 모달
      modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center' },
      modalCard: { width: '86%', maxWidth: 420, backgroundColor: '#252932', borderRadius: 16, padding: 16 },
      modalTitle: { fontFamily: 'Pretendard-SemiBold', fontSize: 16, color: '#EDEDED', marginBottom: 6 },
      modalActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 8, marginTop: 8 },
      modalBtn: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: 10 },
      outline: { backgroundColor: '#2F3440' },
      filled: { backgroundColor: '#2563EB' },
      modalBtnText: { fontFamily: 'Pretendard-Medium', fontSize: 13, color: '#EDEDED' },
      filledText: { color: '#FFFFFF' },
    };
  }, [width]);

  // 상태
  const [logs, setLogs] = useState([]);                 // ✅ 전체 로그
  const [selectedDate, setSelectedDate] = useState(null); // Date | null
  const [showPicker, setShowPicker] = useState(false);
  const [tempDate, setTempDate] = useState(new Date());   // 모달 임시 날짜
  const [sortKey, setSortKey] = useState('prob');         // 'prob' | 'new' | 'old'

  // 포커스될 때마다 최신 로그 로드
  useFocusEffect(
    useCallback(() => {
      let cancelled = false;
      (async () => {
        const rows = await getLogs();
        if (!cancelled) setLogs(rows || []);
      })();
      return () => { cancelled = true; };
    }, [])
  );

  // 날짜별 마킹 데이터: {'YYYY-MM-DD': { any, anyPhish }}
  const marks = useMemo(() => {
    const map = new Map();
    for (const r of logs) {
      const key = r?.dateISO;
      if (!key) continue;
      const cur = map.get(key) || { any: false, anyPhish: false };
      cur.any = true;
      if (Number(r.label) === 1) cur.anyPhish = true;
      map.set(key, cur);
    }
    return map;
  }, [logs]);

  // 필터 + 정렬
  const filtered = useMemo(() => {
    let rows = [...logs];
    if (selectedDate) {
      const target = ymd(selectedDate);
      rows = rows.filter((r) => r.dateISO === target);
    }
    if (sortKey === 'prob') {
      rows.sort((a, b) => (Number(b.prob ?? 0) - Number(a.prob ?? 0)));
    } else if (sortKey === 'new') {
      rows.sort((a, b) => b.createdAt - a.createdAt);
    } else if (sortKey === 'old') {
      rows.sort((a, b) => a.createdAt - b.createdAt);
    }
    return rows;
  }, [logs, selectedDate, sortKey]);

  // 날짜 라벨
  const dateLabel = selectedDate ? ymd(selectedDate) : '날짜 선택';

  const renderCard = ({ item }) => {
    const p = Number.isFinite(item.prob) ? item.prob : null;
    const probPct = p == null ? '-' : (p * 100).toFixed(1) + '%';
    return (
      <View style={s.card}>
        <View style={s.cardHeader}>
          <Text style={s.headText}>날짜</Text>
          <Text style={s.headText}>내용</Text>
          <Text style={s.headText}>확률</Text>
        </View>
        <View style={s.row}>
          <View style={s.dateCol}><Text style={s.dateTextCol}>{item.dateISO}</Text></View>
          <View style={s.contentCol}><Text style={s.contentText}>{item.content}</Text></View>
          <View style={s.probCol}><Text style={s.probText}>{probPct}</Text></View>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={home.screen} edges={['top']}>
      {/* 헤더 */}
      <View style={s.header}>
        <Text style={s.headerTitle}>피싱 로그</Text>
      </View>

      {/* 설명 */}
      <Text style={s.subtitle}>AI가 메시지를 분석한 결과를 여기에 모아뒀어요.</Text>

      {/* 상단 컨트롤 */}
      <View style={s.controls}>
        <Pressable onPress={() => { setTempDate(selectedDate || new Date()); setShowPicker(true); }} style={s.dateIconBtn}>
          <Image source={require('../../assets/icons/calander.png')} style={s.dateIcon} resizeMode="contain" />
        </Pressable>
        <Text style={s.dateText}>{dateLabel}</Text>

        <View style={s.segWrap}>
          {[{ key: 'prob', label: '확률순' }, { key: 'new', label: '최근순' }, { key: 'old', label: '오래된순' }].map(({ key, label }) => {
            const active = sortKey === key;
            return (
              <Pressable key={key} onPress={() => setSortKey(key)} style={[s.segBtn, active && s.segActive]}>
                <Text style={[s.segText, active && s.segActiveText]}>{label}</Text>
              </Pressable>
            );
          })}
        </View>
      </View>

      {/* 날짜 피커 모달 */}
      <Modal visible={showPicker} transparent animationType="fade" onRequestClose={() => setShowPicker(false)}>
        <View style={s.modalOverlay}>
          <View style={s.modalCard}>
            <Text style={s.modalTitle}>날짜 선택</Text>

            {/* 월 날짜 칩 (로그 있는 날 ● 표시) */}
            <View style={{ marginBottom: 8 }}>
              <Text style={{ color: '#EDEDED', marginBottom: 6, fontFamily: 'Pretendard-Medium' }}>로그 있는 날짜</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                {getMonthDays(tempDate).map((d) => {
                  const key = ymd(d);
                  const m = marks.get(key);
                  const isSelected = ymd(tempDate) === key;
                  return (
                    <Pressable
                      key={key}
                      onPress={() => setTempDate(new Date(d.getFullYear(), d.getMonth(), d.getDate()))}
                      style={{
                        width: 40, height: 40, borderRadius: 20, marginRight: 8,
                        alignItems: 'center', justifyContent: 'center',
                        backgroundColor: isSelected ? '#2563EB' : '#2F3440',
                        borderWidth: isSelected ? 0 : 1, borderColor: '#3A3F49',
                      }}
                    >
                      <Text style={{ color: isSelected ? '#fff' : '#E7EAF0', fontFamily: 'Pretendard-Medium' }}>
                        {d.getDate()}
                      </Text>
                      {m?.any && (
                        <View style={{ width: 6, height: 6, borderRadius: 3, marginTop: 2, backgroundColor: m.anyPhish ? '#EF4444' : '#8A909D' }} />
                      )}
                    </Pressable>
                  );
                })}
              </ScrollView>
            </View>

            {/* 시스템 DateTimePicker (월/연 이동용), 날짜는 자정으로 고정 */}
            <DateTimePicker
              value={tempDate}
              mode="date"
              display={Platform.OS === 'ios' ? 'inline' : 'spinner'}
              onChange={(e, d) => {
                if (!d) return;
                const nd = new Date(d.getFullYear(), d.getMonth(), d.getDate());
                setTempDate(nd);
              }}
            />

            <View style={s.modalActions}>
              <Pressable onPress={() => { setSelectedDate(null); setShowPicker(false); }} style={[s.modalBtn, s.outline]}>
                <Text style={s.modalBtnText}>전체</Text>
              </Pressable>
              <Pressable onPress={() => setShowPicker(false)} style={[s.modalBtn, s.outline]}>
                <Text style={s.modalBtnText}>취소</Text>
              </Pressable>
              <Pressable
                onPress={() => {
                  const nd = new Date(tempDate.getFullYear(), tempDate.getMonth(), tempDate.getDate());
                  setSelectedDate(nd);
                  setShowPicker(false);
                }}
                style={[s.modalBtn, s.filled]}
              >
                <Text style={[s.modalBtnText, s.filledText]}>적용</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* 리스트 */}
      {filtered.length === 0 ? (
        <View style={s.empty}><Text style={s.emptyText}>저장된 로그가 없습니다.</Text></View>
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(item) => item.id ?? String(item.createdAt)}
          contentContainerStyle={s.list}
          renderItem={renderCard}
        />
      )}
    </SafeAreaView>
  );
}
