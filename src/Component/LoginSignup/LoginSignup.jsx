import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import "./LoginSignup.css";

import user_icon from "../Assets/person.png";
import email_icon from "../Assets/email.png";
import password_icon from "../Assets/password.png";
import { useUserContext } from "../../LocalData/UserContext";

const LoginSignup = () => {
  const [action, setAction] = useState("Sign Up");

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const { dataUser, setDataUser } = useUserContext();
  const { setIsLogin } = useUserContext();
  const { setDataUserLogin } = useUserContext();
  useEffect(() => {
    //  tra dataUser đã đc thêm mới (không xoá)
    console.log("New dataUser:", dataUser);
  }, [dataUser]);
  const signUpUser = () => {
    setDataUser([
      ...dataUser,
      { name: name, email: email, password: password },
    ]);
    alert("Đăng ký tài khoản thành công!");
  };

  const login = () => {
    dataUser.forEach((element) => {
      if (element.email == email && element.password == password) {
        setIsLogin(true);
        setDataUserLogin(element);
        alert("Đăng nhập tài khoản thành công!");
      }
    });
  };

  return (
    <div className="container">
      <div className="header">
        <div className="text">{action}</div>
        <div className="underline"></div>
      </div>
      <div className="inputs">
        {action === "Sign Up" && (
          <div className="input">
            <img src={user_icon} alt="" />
            <input
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
              }}
            />
          </div>
        )}
        <div className="input">
          <img src={email_icon} alt="" />
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value.trim());
            }}
          />
        </div>
        <div className="input">
          <img src={password_icon} alt="" />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value.trim());
            }}
          />
        </div>
      </div>
      <div className="forgot-password">
        Lost Password? <span>Click here</span>
      </div>
      <div className="submit-container">
        <div
          className={action === "Sign Up" ? "submit " : "submit gray"}
          onClick={() => {
            if (action === "Sign Up") {
              signUpUser();
            }
            setAction("Sign Up");
          }}
        >
          Sign Up
        </div>
        <div
          className={action === "Login" ? "submit" : "submit gray"}
          onClick={() => {
            if (action === "Login") {
              login();
            }
            setAction("Login");
          }}
        >
          Login
        </div>
      </div>
    </div>
  );
};
export default LoginSignup;
