
import logging
import os
import datetime
import traceback

def save_successful_run(args, file='run_log.txt'):
    run_log_path = os.path.join('./', file)
    os.makedirs('./', exist_ok=True)
    with open(run_log_path, 'a') as f:
        f.write("\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Name: {args.dataset}_{args.model_name}\n")
        # f.write(f"Row Column: {args.row_column}\n")  # Assuming args.row_column exists
        f.write("Run was successful.\n")

def save_error_details_to_file(args, error):
    error_log_path = os.path.join(args.save_path, 'error_log.txt')
    os.makedirs(args.save_path, exist_ok=True)
    with open(error_log_path, 'a') as f:
        f.write("\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error: {str(error)}\n")
        f.write("Stack Trace:\n")
        traceback.print_exc(file=f)

def save_args_to_file(args, command, log_dir=''):
    args_file_path = os.path.join(args.save_path, log_dir, 'args.txt')
    os.makedirs(os.path.dirname(args_file_path), exist_ok=True)
    if os.path.exists(args_file_path):
        print(f"Warning: The file {args_file_path} already exists and will be overwritten.")
    with open(args_file_path, 'a') as f:  # Change 'w' to 'a' to append to the file
        f.write("\n")  # Add new line before writing to the file
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # Add timestamp
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"Command arguments: {' '.join(command)}\n")  # Add the command arguments to the file

def get_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger