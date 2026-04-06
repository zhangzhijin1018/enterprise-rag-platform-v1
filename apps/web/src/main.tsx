/* React 应用入口。负责把根组件挂载到 index.html 中的 #root。 */

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// StrictMode 只在开发期帮助发现副作用问题，不改变生产行为。
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
