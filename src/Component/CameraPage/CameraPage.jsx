import React, { useState, useEffect, useRef } from "react";
import Papa from "papaparse";
import "./CameraPage.css";

import user_icon from "../Assets/person.png";
import email_icon from "../Assets/email.png";
import password_icon from "../Assets/password.png";
import { useUserContext } from "../../LocalData/UserContext";
import axios from "axios";

const CameraPage = () => {
  const { dataUserLogin } = useUserContext();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isUsing, setIsUsing] = useState(false);

  const sizeImage = 300;

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    startCamera();
  }, []);

  const captureImage = async () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      // Đặt kích thước canvas bằng với kích thước video
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      // Vẽ hình ảnh từ video lên canvas
      context.drawImage(
        videoRef.current,
        0,
        0,
        videoRef.current.videoWidth,
        videoRef.current.videoHeight,
        0,
        0,
        canvas.width,
        canvas.height
      );

      //Lấy hình ảnh từ canvas với định dạng base64 và truyền đi ( không cần xử lí chuỗi chỗ này nữa)
      const imageData64 = canvas.toDataURL("image/png").split(",")[1];

      // Gửi hình ảnh lên server
      try {
        axios
          .post("http://127.0.0.1:5000/eyes", {
            data: imageData64,
          })
          .then(function (response) {
            captureImage();
            document.getElementById("imageContainer").innerHTML = `<img src="${
              "data:image/jpeg;base64," + response.data.data
            }" alt="Returned Image" />`;
          })
          .catch(function (error) {
            console.log(error);
          });
      } catch (error) {
        console.error("Error sending image to server:", error);
      }
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="text">Welcom {dataUserLogin.name}</div>
        <div className="underline"></div>
        <canvas ref={canvasRef} style={{ display: "none" }} />
        <video ref={videoRef} autoPlay playsInline />
        <div id="imageContainer"></div>
        <button
          className="submit"
          onClick={() => {
            setIsUsing(!isUsing);
            captureImage();
          }}
        >
          {isUsing ? "Stop Capture Camera" : "Capture Camera"}
        </button>
      </div>
    </div>
  );
};
export default CameraPage;
