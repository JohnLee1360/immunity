"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";

const Login = () => {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");

  const handleLogin = () => {
    // 这里可以添加登录逻辑，例如发送请求到后端进行验证
    console.log("Logging in with", { username, password });
  };

  return (
    <div className="flex flex-col gap-4">
      <input
        type="text"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="用户名"
        className="border p-2"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="密码"
        className="border p-2"
      />
      <Button onClick={handleLogin} className="mt-4">
        登录
      </Button>
    </div>
  );
};

export default Login;