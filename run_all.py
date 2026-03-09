#!/usr/bin/env python3
"""
Main script to run all AI6103 experiments
Usage: python run_all.py [--section SECTION]
"""
import argparse
import os
import json
import smtplib
import time
from email.mime.text import MIMEText
from email.header import Header
from config import OUTPUT_DIR
from data import compute_dataset_stats, get_dataloaders
from experiments import (
    run_section2, run_section3, run_section4,
    run_section5, run_section6, run_all_experiments
)


def send_email_notification(subject, content):
    """Send email notification with experiment results."""
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf"
    to_email = "chenshunzhang823@gmail.com"

    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = f"AI6103 Server <{sender_email}>"
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 邮件通知已发送")
        return True
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")
        return False


def collect_all_results():
    """Collect all experiment results from output directory."""
    results = {}

    # Collect results from each section
    result_files = [
        ('section3_results.json', 'Learning Rate Experiments'),
        ('section4_results.json', 'Learning Rate Schedule'),
        ('section5_results.json', 'Weight Decay'),
        ('section6_results.json', 'Mixup Augmentation'),
    ]

    for filename, section_name in result_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[section_name] = data

    return results


def format_results_email(results, total_time=None):
    """Format results into email content."""
    lines = [
        "【AI6103 超参数研究实验报告】",
        "=" * 50,
        f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if total_time:
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        lines.append(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")

    lines.append("")

    for section_name, section_results in results.items():
        lines.append(f"\n【{section_name}】")
        lines.append("-" * 40)

        for exp_name, exp_data in section_results.items():
            final_train_acc = exp_data.get('final_train_acc', 0)
            final_val_acc = exp_data.get('final_val_acc', 0)
            final_train_loss = exp_data.get('final_train_loss', 0)
            final_val_loss = exp_data.get('final_val_loss', 0)

            lines.append(f"  {exp_name}:")
            lines.append(f"    Train Acc: {final_train_acc:.2f}%")
            lines.append(f"    Val Acc: {final_val_acc:.2f}%")
            lines.append(f"    Train Loss: {final_train_loss:.4f}")
            lines.append(f"    Val Loss: {final_val_loss:.4f}")

    # Find best overall result
    best_acc = 0
    best_exp = ""
    for section_name, section_results in results.items():
        for exp_name, exp_data in section_results.items():
            val_acc = exp_data.get('final_val_acc', 0)
            if val_acc > best_acc:
                best_acc = val_acc
                best_exp = f"{section_name} - {exp_name}"

    if best_exp:
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"🏆 最佳结果: {best_exp}")
        lines.append(f"   验证准确率: {best_acc:.2f}%")

    lines.append("")
    lines.append(f"结果文件目录: {OUTPUT_DIR}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='AI6103 Hyperparameter Study')
    parser.add_argument('--section', type=int, choices=[2, 3, 4, 5, 6],
                        help='Run only a specific section')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset first')
    parser.add_argument('--no-email', action='store_true',
                        help='Skip email notification')
    args = parser.parse_args()

    # Download data if requested
    if args.download:
        from download_data import download_food11
        download_food11()
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    if args.section:
        # Run specific section
        # First get dataset stats
        from config import DATA_DIR
        train_path = os.path.join(DATA_DIR, 'training')
        mean, std = compute_dataset_stats(train_path)

        if args.section == 2:
            run_section2()
        elif args.section == 3:
            run_section3(mean, std)
        elif args.section == 4:
            # Need best LR from section 3
            best_lr_name = "lr_0.025"  # Default, should be updated from section 3 results
            run_section4(mean, std, best_lr_name)
        elif args.section == 5:
            best_lr_name = "lr_0.025"
            run_section5(mean, std, best_lr_name)
        elif args.section == 6:
            best_lr_name = "lr_0.025"
            run_section6(mean, std, best_lr_name)
    else:
        # Run all experiments
        run_all_experiments()

    # Send email notification with results
    if not args.no_email:
        total_time = time.time() - start_time
        results = collect_all_results()

        if results:
            email_content = format_results_email(results, total_time)
            subject = f"AI6103 实验完成 - {time.strftime('%Y-%m-%d %H:%M')}"
            send_email_notification(subject, email_content)
        else:
            print("⚠️ 未找到实验结果，跳过邮件通知")


if __name__ == "__main__":
    main()
