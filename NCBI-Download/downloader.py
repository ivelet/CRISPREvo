from helpers import download_from_file_multi
from settings import use_input_file, use_output_folder, use_number_chunks, use_db_type, use_rettype, use_number_lines


def main():
    download_from_file_multi(use_input_file, use_output_folder, use_number_chunks, use_db_type, use_rettype, use_number_lines)


if __name__ == "__main__":
    main()






