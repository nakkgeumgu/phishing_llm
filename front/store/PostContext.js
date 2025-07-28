import React, { createContext, useState, useEffect } from 'react';

export const PostContext = createContext();

export const PostProvider = ({children}) => {
  const [posts, setPosts] = useState([]);
  
  useEffect(() => {
    const fetchPosts = async () => {
      const dummy = [
        { id: 1, title: '첫 번째 글', content: '이것은 첫 번째 글의 내용입니다.', author: '작성자1', uid: '1'},
        { id: 2, title: '두 번째 글', content: '이것은 두 번째 글의 내용입니다.', author: '작성자2', uid: '2'},
      ];
      setPosts(dummy);
    };
    fetchPosts();
  }, []);

  return (
    <PostContext.Provider value={{posts, setPosts}}>
      {children}
    </PostContext.Provider>
  );
}