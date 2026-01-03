import shutil

print("Đang cập nhật giao diện mới...")

# Copy base_new.html -> base.html
shutil.copy("templates/base_new.html", "templates/base.html")
print("✓ Đã cập nhật base.html")

# Copy index_new.html -> index.html  
shutil.copy("templates/index_new.html", "templates/index.html")
print("✓ Đã cập nhật index.html")

print("\n✅ Hoàn thành! Hãy refresh trình duyệt (F5 hoặc Ctrl+R)")
print("URL: http://127.0.0.1:8000/")
