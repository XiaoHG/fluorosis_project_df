#!/usr/bin/env python3
"""发送项目整理报告到指定邮箱 (Gmail SMTP)."""

import smtplib
import os
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_report(report_text: str, to_email: str = "xiaohggg@gmail.com",
                from_email: str = None, app_password: str = None) -> bool:
    """通过 Gmail SMTP 发送整理报告."""
    from_email = from_email or os.environ.get("GMAIL_USER", "xiaohggg@gmail.com")
    app_password = app_password or os.environ.get("GMAIL_APP_PASSWORD", "")

    if not app_password:
        print("ERROR: 请设置环境变量 GMAIL_APP_PASSWORD")
        print("获取 App Password: https://myaccount.google.com/apppasswords")
        return False

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = f"氟斑牙项目整理报告 — {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(report_text, "plain", "utf-8"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.starttls()
        server.login(from_email, app_password)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
        print(f"报告已发送至 {to_email}")
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False


if __name__ == "__main__":
    if sys.stdin.isatty():
        print("Usage: python send_report.py < report.txt")
        sys.exit(1)

    report = sys.stdin.read()
    if not report.strip():
        print("ERROR: 报告内容为空")
        sys.exit(1)

    send_report(report)
