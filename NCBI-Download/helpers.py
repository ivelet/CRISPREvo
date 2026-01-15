from Bio import Entrez
import os
from os.path import isfile, join
from os import walk
from multiprocessing import Pool


def create_n_internal_folders(folder_name, n):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for i in range(1, n+1):
        folder_name_i = join(folder_name, str(i))
        if not os.path.exists(folder_name_i):
            os.makedirs(folder_name_i)


def assign_chunk_to_index(total_numbers, number_chunks):
    chunk_size = total_numbers // number_chunks
    residual = total_numbers % number_chunks
    chunk_sizes = [chunk_size + 1 if (i <= residual - 1) else chunk_size for i in range(number_chunks)]
    cumulative_sizes = [sum(chunk_sizes[:i]) for i in range(number_chunks+1)]

    dict_chunk_sizes = {}
    for i in range(1, total_numbers+1):
        for j in cumulative_sizes:
            if i <= j:
                chunk_index = cumulative_sizes.index(j)
                dict_chunk_sizes[i] = chunk_index
                break

    return dict_chunk_sizes


def list_files_in_folder_recursive(folder):
    all_files = []
    for root, dirs, files in walk(folder):
        for file in files:
            all_files.append(file.split(".")[0])
    return all_files


def download_file(file_name, folder, db_type, rettype, num_lines=None):
    print("Downloading", file_name)
    for i in range(10):
        try:
            Entrez.email = "Your.Name.Here@example.org"

            handle = Entrez.efetch(db=db_type, id=file_name, rettype=rettype, retmode="text")
            full_file_name = join(folder, file_name + f".{rettype}")
            if not num_lines:
                header_line = handle.readline()
                with open(full_file_name, 'w') as local_file:
                    local_file.write(header_line)
                    local_file.write(handle.read())
                handle.close()
            else:
                with open(full_file_name, 'w') as local_file:
                    for i in range(num_lines):
                        local_file.write(handle.readline())
            break
        except Exception:
            pass


def download_from_file(input_file, folder_name, number_chunks=1, db_type="nucleotide", rettype="fasta", num_lines=None):
    existing_files = list_files_in_folder_recursive(folder_name)

    with open(input_file, "r") as f:
        lines = f.readlines()

    all_files = [line.strip().split("\t")[0] for line in lines]
    all_files = [x.split(".")[0] for x in all_files]
    all_files = list(set(all_files))

    if number_chunks == 1:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for index, file_name in enumerate(all_files, 1):
            print(index, file_name, len(all_files))
            if file_name in existing_files:
                continue
            if "." in file_name:
                file_name = file_name.split(".")[0]
            if "Contig" in file_name:
                continue
            download_file(file_name, folder_name, db_type, rettype, num_lines)
    else:
        create_n_internal_folders(folder_name, number_chunks)
        index_to_chunk = assign_chunk_to_index(len(all_files), number_chunks)
        for index, file_name in enumerate(all_files, 1):
            print(index, file_name, len(all_files))
            if file_name in existing_files:
                continue
            if "." in file_name:
                file_name = file_name.split(".")[0]
            if "Contig" in file_name:
                continue
            chunk_to_assign = index_to_chunk[index]
            folder_to_assign = join(folder_name, str(chunk_to_assign))
            download_file(file_name, folder_to_assign, db_type, rettype, num_lines)


def download_from_file_multi(input_file, folder_name, number_chunks=1, db_type="nucleotide",
                             rettype="fasta",num_lines=None):
    existing_files = list_files_in_folder_recursive(folder_name)

    with open(input_file, "r") as f:
        lines = f.readlines()

    all_files = [line.strip().split("\t")[0] for line in lines]
    all_files = [x.split(".")[0] for x in all_files]
    all_files = list(set(all_files))

    if number_chunks == 1:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for index, file_name in enumerate(all_files, 1):
            print(index, file_name, len(all_files))
            if file_name in existing_files:
                continue
            if "." in file_name:
                file_name = file_name.split(".")[0]
            if "Contig" in file_name:
                continue
            download_file(file_name, folder_name, db_type, rettype)
    else:
        create_n_internal_folders(folder_name, number_chunks)
        index_to_chunk = assign_chunk_to_index(len(all_files), number_chunks)
        print(index_to_chunk)
        full_folder_paths = [join(folder_name, str(value)) for key, value in index_to_chunk.items()]
        list_inputs = [(file_name, folder_to_assign, db_type, rettype, num_lines) for file_name, folder_to_assign in zip(all_files, full_folder_paths)]
        print("start multiprocessing download")
        for element in list_inputs:
            print(element)
        with Pool(10) as p:
            p.starmap(download_file, list_inputs)


