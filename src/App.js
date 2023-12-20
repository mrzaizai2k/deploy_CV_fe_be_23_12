import logo from "./logo.svg";
import "./App.css";
import LoginSignup from "./Component/LoginSignup/LoginSignup";
import {
  UserContextProvider,
  useUserContext,
} from "../src/LocalData/UserContext";
import { useState } from "react";
import CameraPage from "./Component/CameraPage/CameraPage";

function App() {
  return (
    <UserContextProvider>
      <AppContent />
    </UserContextProvider>
  );
}

function AppContent() {
  const { isLogin, setIsLogin } = useUserContext();

  return <div>{!isLogin ? <LoginSignup /> : <CameraPage />}</div>;
}

export default App;
