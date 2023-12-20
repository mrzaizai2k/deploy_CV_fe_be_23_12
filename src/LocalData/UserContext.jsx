import React, { createContext, useContext, useState } from "react";

const UserContext = createContext();

export const UserContextProvider = ({ children }) => {
  const [dataUser, setDataUser] = useState([
    { name: "admin", email: "admin@gmail.com", password: "123456789" },
  ]);
  const [isLogin, setIsLogin] = useState(false);
  const [dataUserLogin, setDataUserLogin] = useState();

  return (
    <UserContext.Provider
      value={{
        dataUser,
        setDataUser,
        isLogin,
        setIsLogin,
        dataUserLogin,
        setDataUserLogin,
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export const useUserContext = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error("useUserContext must be used within a UserContextProvider");
  }
  return context;
};
