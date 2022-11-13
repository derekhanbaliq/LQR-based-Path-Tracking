import xlsxwriter


def write_xlsx(steering):
    workbook = xlsxwriter.Workbook('steering_speed.xlsx')
    worksheet = workbook.add_worksheet()

    row = 0
    column = 0

    # iterating through content list
    for item in steering:

        # write operation perform
        worksheet.write(row, column, item)

        # incrementing the value of row by one
        # with each iterations.
        row += 1

    workbook.close()
