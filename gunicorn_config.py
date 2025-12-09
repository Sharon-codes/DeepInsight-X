# gunicorn_config.py
import os

# Number of worker processes. A common recommendation is (2 * CPU_CORES) + 1.
# Adjust based on your HPC node's CPU cores and memory.
workers = int(os.environ.get('GUNICORN_PROCESSES', '4')) 
# Number of threads per worker. Use threads if your application is I/O bound.
threads = int(os.environ.get('GUNICORN_THREADS', '2'))   
# Bind to a Unix socket for Nginx to proxy requests. This is more efficient than TCP for local communication. [30, 31]
bind = os.environ.get('GUNICORN_BIND', 'unix:/tmp/medical_xray_app.sock') 

# Timeout for workers. Increase if your model inference takes a long time. [32]
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120')) # 120 seconds

# Log level
loglevel = os.environ.get('GUNICORN_LOGLEVEL', 'info') # 'debug', 'info', 'warning', 'error', 'critical' [30]

# Access log file
accesslog = os.environ.get('GUNICORN_ACCESSLOG', '-') # '-' means stdout [30]
# Error log file
errorlog = os.environ.get('GUNICORN_ERRORLOG', '-') # '-' means stderr [30]

# Recommended for passing protocol information (e.g., for HTTPS if Nginx handles SSL) [32, 30, 31]
secure_scheme_headers = {
    'X-Forwarded-Proto': 'https'
}

# If Nginx is on a different host, specify trusted IPs.
# For a single node deployment with Nginx and Gunicorn on the same machine, this might not be strictly needed
# but good practice if you have a proxy. [30, 31]
# forwarded_allow_ips = '*' # Potentially dangerous if untrusted connections are possible
# forwarded_allow_ips = '127.0.0.1' # Trust localhost